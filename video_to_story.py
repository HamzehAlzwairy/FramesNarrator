import numpy as np
import skipthoughts
import numpy
import decoder
import nltk
nltk.download('punkt')

def generate(method, skip_thought_vectors, z,):

    if method == 'mean':
        shifted_vector = skip_thought_vectors.mean(0) - z['bneg'] + z['bpos']
        # Generate story conditioned on shift
        passage = decoder.run_sampler(z['dec'], shifted_vector, beam_width=5)
        print(passage)
    elif method == 'seq':
        for vec in skip_thought_vectors:
            shifted_vector = vec - z['bneg'] + z['bpos']
            # Generate story conditioned on shift
            passage = decoder.run_sampler(z['dec'], shifted_vector, beam_width=5)
            print(passage)
    elif method == 'seq_self':
        for i, vec in enumerate(skip_thought_vectors):
            shifted_vector = vec - z['bneg'] + skip_thought_vectors.mean(0)
            # Generate story conditioned on shift
            passage = decoder.run_sampler(z['dec'], shifted_vector, beam_width=5)
            print(f'S{i}: ',passage)
    elif method == 'interpolate':
        alpha = 0.2
        for i, (vec1, vec2) in enumerate(zip(skip_thought_vectors[:-1], skip_thought_vectors[1:])):
            for a in np.linspace(0, 1, (1 / alpha)+1)[:-1]:
                vec = vec1 * (1-a) + vec2 * a
                shifted_vector = vec - z['bneg'] + z['bpos']
                # Generate story conditioned on shift
                passage = decoder.run_sampler(z['dec'], shifted_vector, beam_width=5)
                passage = passage.split('.')[0]
                print(f"[{1-a:.1f} * S{i} + {a:.1f} * S{i+1}]",passage)


if __name__ == '__main__':

    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)

    decoder_model_path = "./data/romance.npz"
    decoder_dictionary_path = "./data/romance_dictionary.pkl"
    romance_style_path = "./data/romance_style.npy"
    caption_style_path = "./data/caption_style.npy"

    # sentences = ["the dragon on the roof.",
    #              "the couple on the set of tv drama."]

    sentences = ["the group of people in the forest",
                 'a woman is lying on a tree branch in the forest',
                 "person is a talented archer, but he is also a very dangerous man",
                 'a man lying in the forest',
                 'the boys are running in the forest']
    # Decoder
    print('Loading decoder...')
    dec = decoder.load_model(decoder_model_path, decoder_dictionary_path)

    # Biases
    print('Loading biases...')
    bneg = numpy.load(caption_style_path)
    bpos = numpy.load(romance_style_path)

    z = {}
    z['dec'] = dec
    z['bneg'] = bneg
    z['bpos'] = bpos

    print('Input Sentences:')
    print(sentences)
    skip_thought_vectors = encoder.encode(sentences)

    print('Decoding with various style shifting methods')

    print('method: mean')
    method ='mean'
    generate(method, skip_thought_vectors, z)

    print('method: seq_self')
    method ='seq_self'
    generate(method, skip_thought_vectors, z)

    print('method: interpolate')
    method ='interpolate'
    generate(method, skip_thought_vectors, z)
