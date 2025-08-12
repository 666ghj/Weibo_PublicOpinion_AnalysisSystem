from bertopic import BERTopic


def test_load_save_model():
    model = BERTopic(language="Dutch", embedding_model=None)
    model.save("test", serialization="pickle")
    loaded_model = BERTopic.load("test")
    assert type(model) is type(loaded_model)
    assert model.language == loaded_model.language
    assert model.embedding_model == loaded_model.embedding_model
    assert model.top_n_words == loaded_model.top_n_words


def test_get_params():
    model = BERTopic()
    params = model.get_params()
    assert not params["embedding_model"]
    assert not params["low_memory"]
    assert not params["nr_topics"]
    assert params["n_gram_range"] == (1, 1)
    assert params["min_topic_size"] == 10
    assert params["language"] == "english"
