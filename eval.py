import senteval


def prepare(params, samples):
  """
    In this example we are going to load Glove, 
    here you will initialize your model.
    remember to add what you model needs into the params dictionary
    """
  _, params.word2id = data.create_dictionary(samples)
  # load glove/word2vec format
  params.word_vec = data.get_wordvec(PATH_TO_VEC, params.word2id)
  # dimensionality of glove embeddings
  params.wvec_dim = 300
  return


def batcher(params, batch):
  """
    In this example we use the average of word embeddings as a sentence representation.
    Each batch consists of one vector for sentence.
    Here you can process each sentence of the batch, 
    or a complete batch (you may need masking for that).
    
    """
  # if a sentence is empty dot is set to be the only token
  # you can change it into NULL dependening in your model
  batch = [sent if sent != [] else ['.'] for sent in batch]
  embeddings = []

  for sent in batch:
    sentvec = []
    # the format of a sentence is a lists of words (tokenized and lowercased)
    for word in sent:
      if word in params.word_vec:
        # [number of words, embedding dimensionality]
        sentvec.append(params.word_vec[word])
    if not sentvec:
      vec = np.zeros(params.wvec_dim)
      # [number of words, embedding dimensionality]
      sentvec.append(vec)
    # average of word embeddings for sentence representation
    # [embedding dimansionality]
    sentvec = np.mean(sentvec, 0)
    embeddings.append(sentvec)
  # [batch size, embedding dimensionality]
  embeddings = np.vstack(embeddings)
  return embeddings


# Set params for SentEval
# we use logistic regression (usepytorch: Fasle) and kfold 10
# In this dictionary you can add extra information that you model needs for initialization
# for example the path to a dictionary of indices, of hyper parameters
# this dictionary is passed to the batched and the prepare fucntions
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 10}
# this is the config for the NN classifier but we are going to use scikit-learn logistic regression with 10 kfold
# usepytorch = False
#params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
#                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
  se = senteval.engine.SE(params_senteval, batcher, prepare)

  # here you define the NLP taks that your embedding model is going to be evaluated
  # in (https://arxiv.org/abs/1802.05883) we use the following :
  # SICKRelatedness (Sick-R) needs torch cuda to work (even when using logistic regression),
  # but STS14 (semantic textual similarity) is a similar type of semantic task
  transfer_tasks = [
      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC', 'SICKEntailment',
      'STS14'
  ]
  # senteval prints the results and returns a dictionary with the scores
  results = se.eval(transfer_tasks)
  print(results)
