"""
CellWhisperer embedding model server
====================================

Used as micro-service to prevent model loading for each dataset-specific servers
"""


from cellwhisperer.utils.inference import score_transcriptomes_vs_texts
from cellwhisperer.utils.model_io import load_cellwhisperer_model
from cellwhisperer.config import get_path, config

from flask import Flask, request, Response
import pickle

BATCH_SIZE = 128
MODEL_PATH = (
    get_path(["paths", "jointemb_models"])
    / f"{config['model_name_path_map']['cellwhisperer']}.ckpt"
)

app = Flask(__name__)

# load model
pl_model, tokenizer, transcriptome_processor = load_cellwhisperer_model(MODEL_PATH)


@app.route("/api/text_embedding", methods=["POST"])
def text_api_endpoint():
    """
    Takes a (json) list of strings and returns their embeddings as np array
    """
    data = request.get_json()
    text_embeds = pl_model.model.embed_texts(data)

    # Serialize the result
    return Response(
        pickle.dumps(text_embeds.cpu().numpy()), mimetype="application/octet-stream"
    )


@app.route("/api/score_transcriptomes_vs_texts", methods=["POST"])
def scoring_api_endpoint():
    """
    Note: this is slow because of the large ize of transcriptome_embeds (transfer time)

    """
    data = request.get_data()
    # Deserialize the input data
    (
        transcriptome_embeds,
        text_list,
        average_mode,
        transcriptome_annotations,
        score_norm_method,
    ) = pickle.loads(data)
    # Call your function
    scores, _ = score_transcriptomes_vs_texts(
        transcriptome_embeds.to(pl_model.model.device),
        text_list,
        pl_model.model.discriminator.temperature.exp(),
        pl_model.model,
        average_mode,
        transcriptome_annotations,
        transcriptome_processor,  # unused
        BATCH_SIZE,
        score_norm_method,
    )
    # Serialize the result
    return Response(pickle.dumps(scores), mimetype="application/octet-stream")


@app.route("/api/store_cache", methods=["POST"])
def store_cache_endpoint():
    pl_model.model.store_cache()


@app.route("/api/logit_scale", methods=["GET"])
def get_logit_scale_endpoint():
    logit_scale = pl_model.model.discriminator.temperature.exp().item()
    # return the float as a single value in the content
    return Response(str(logit_scale))


if __name__ == "__main__":
    app.run(debug=False, port=8910, host="0.0.0.0")
