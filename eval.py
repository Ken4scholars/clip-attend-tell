import copy
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from clip import clip

from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import torch.nn.functional as F
from tqdm import tqdm

# Parameters
data_folder = 'flickr'  # folder with data files saved by create_input_files.py
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint = 'models/BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
clip_checkpoint = None
word_map_file = 'flickr/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
use_clip = True
clip_beam_search = True

# Load model
if not clip_checkpoint:
    clip_checkpoint = str(checkpoint[-8:]) + '_clip.pt'
checkpoint = torch.load(checkpoint, map_location=device)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
if os.path.exists(clip_checkpoint) and hasattr(encoder, 'clip_model'):
    encoder.load_clip_from_disk(clip_checkpoint)
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    _transforms = [normalize]
    if use_clip:
        _, preprocess = clip.load('ViT-B/32')
        preprocess.transforms = preprocess.transforms[:2]
        _transforms = preprocess.transforms + _transforms
    _transforms = transforms.Compose(_transforms)
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=_transforms),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        rev_word_map = {v: k for k, v in word_map.items()}
        if clip_beam_search:
            with torch.no_grad():
                image_features = encoder.clip_model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)

        def get_clip_scores(seqs, scores):
            nonlocal top_k_scores
            special_words = ['<start>', '<end>']
            replace_words = {'<unk>': '<averyunpleasantword>', '<pad>': '<anotherveryunpleasantword>'}
            special_words_enc = [word_map[w] for w in special_words]
            if step == 1:
                top_k_scores, next_word_inds = scores[0].topk(k, 0, True, True)  # (s)
                return torch.zeros(k, device=device).long(), next_word_inds
            next_word_inds = scores.topk(k)[1]
            inds = []

            text = []
            weights = torch.ones(k ** 2).to(device)
            count = 0
            for idx, (prev_seq, next_words) in enumerate(zip(seqs.tolist(), next_word_inds.tolist())):
                prev_words = [rev_word_map[w] for w in prev_seq if w not in special_words_enc]
                for word in next_words:
                    cap_words = copy.copy(prev_words)
                    if word not in special_words:
                        word_char = rev_word_map[word]
                        word_char = replace_words.get(word_char) or word_char
                        cap_words.append(word_char)
                    text.append(' '.join(cap_words))
                    inds.append([idx, word])
                    if rev_word_map[word] == '<end>':
                        weights[count] = 1.5
                    count += 1
            inds = np.array(inds)
            text = clip.tokenize(text).to(device)
            with torch.no_grad():
                text_features = encoder.clip_model.encode_text(text)

            # Pick the top k most similar captions for the image
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T * weights).log_softmax(dim=-1)
            top_k_scores, indices = similarity.view(-1).topk(k, 0, True, True)
            prev_inds = torch.tensor([inds[idx][0] for idx in indices], device=device)
            next_inds = torch.tensor([inds[idx][1] for idx in indices], device=device)

            return prev_inds, next_inds

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            if clip_beam_search:
                prev_word_inds, next_word_inds = get_clip_scores(seqs, scores)
            else:

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = (top_k_words / vocab_size).long()  # (s)
                next_word_inds = (top_k_words % vocab_size).long()  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly
            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        if len(complete_inds) > 0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        else:
            i = top_k_scores.argmax().item()
            seq = seqs[i].tolist()

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [rev_word_map[w] for w in c if
                           w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append(
            [rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

    bleu4 = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)

    return bleu4


if __name__ == '__main__':
    beam_size = 5
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))
