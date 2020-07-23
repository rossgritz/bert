import run_classifier as rc 

class FinancialProc(rc.DataProcessor):
  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      n = 0
      if set_type == "train":
      	n = 2222
      elif set_type == "dev":
      	n = 3333
      else:
      	n = 4444
      uid = str(n)+str(i)
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(uid))
      text_a = tokenization.convert_to_unicode(line[0])
      #text_b = tokenization.convert_to_unicode(line[9])
      label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, label=label))
    return examples