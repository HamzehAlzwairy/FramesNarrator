import skipthoughts
import numpy
import decoder
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
sentences = ["A group of friends playing basketball", "A man throwing the ball towards the basket", "Basketball players jumping to snatch the ball"
             ,"The basketball team is enjoying the game"]

decoder_model_path = "../../neural_storyteller/romance.npz"
decoder_dictionary_path = "../../neural_storyteller/romance_dictionary.pkl"
romance_style_path= "../../neural_storyteller/romance_style.npy"
caption_style_path= "../../neural_storyteller/caption_style.npy"

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

print("encoding sentences to st..")
skip_thought_vectors = encoder.encode(sentences)

print("applying style shifting..")
# Style shifting
shifted_vector = skip_thought_vectors.mean(0) - z['bneg'] + z['bpos']

print("decoding st to passage..")
# Generate story conditioned on shift
passage = decoder.run_sampler(z['dec'], shifted_vector, beam_width=50)
print(passage)
