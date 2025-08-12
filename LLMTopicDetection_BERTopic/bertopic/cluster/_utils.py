import numpy as np


def hdbscan_delegator(model, func: str, embeddings: np.ndarray = None):
    """Function used to select the HDBSCAN-like model for generating
    predictions and probabilities.

    Arguments:
        model: The cluster model.
        func: The function to use. Options:
                - "approximate_predict"
                - "all_points_membership_vectors"
                - "membership_vector"
        embeddings: Input embeddings for "approximate_predict"
                    and "membership_vector"
    """
    try:
        import hdbscan
    except (ImportError, ModuleNotFoundError):
        hdbscan = type("hdbscan", (), {"HDBSCAN": None})()

    # Approximate predict
    if func == "approximate_predict":
        if isinstance(model, hdbscan.HDBSCAN):
            predictions, probabilities = hdbscan.approximate_predict(model, embeddings)
            return predictions, probabilities

        str_type_model = str(type(model)).lower()
        if "cuml" in str_type_model and "hdbscan" in str_type_model:
            from cuml.cluster import hdbscan as cuml_hdbscan

            predictions, probabilities = cuml_hdbscan.approximate_predict(model, embeddings)
            return predictions, probabilities

        predictions = model.predict(embeddings)
        return predictions, None

    # All points membership
    if func == "all_points_membership_vectors":
        if isinstance(model, hdbscan.HDBSCAN):
            return hdbscan.all_points_membership_vectors(model)

        str_type_model = str(type(model)).lower()
        if "cuml" in str_type_model and "hdbscan" in str_type_model:
            from cuml.cluster import hdbscan as cuml_hdbscan

            return cuml_hdbscan.all_points_membership_vectors(model)

        return None

    # membership_vector
    if func == "membership_vector":
        if isinstance(model, hdbscan.HDBSCAN):
            probabilities = hdbscan.membership_vector(model, embeddings)
            return probabilities

        str_type_model = str(type(model)).lower()
        if "cuml" in str_type_model and "hdbscan" in str_type_model:
            from cuml.cluster import hdbscan as cuml_hdbscan

            probabilities = cuml_hdbscan.membership_vector(model, embeddings)
            return probabilities

        return None


def is_supported_hdbscan(model):
    """Check whether the input model is a supported HDBSCAN-like model."""
    try:
        import hdbscan
    except (ImportError, ModuleNotFoundError):
        hdbscan = type("hdbscan", (), {"HDBSCAN": None})()

    if isinstance(model, hdbscan.HDBSCAN):
        return True

    str_type_model = str(type(model)).lower()
    if "cuml" in str_type_model and "hdbscan" in str_type_model:
        return True

    return False
