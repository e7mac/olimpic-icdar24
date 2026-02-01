"""
Cog predictor for Zeus OMR model.

Zeus is an Optical Music Recognition (OMR) model that converts images of
pianoform sheet music into Linearized MusicXML (LMX) format.
"""

import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Add app directory to path for imports
sys.path.insert(0, "/src")

from typing import Union

import numpy as np
import tensorflow as tf
from cog import BasePredictor, Input, Path as CogPath

from app.linearization.Delinearizer import Delinearizer
from app.symbolic.part_to_score import part_to_score

# Model directory path (downloaded during build)
MODEL_DIR = "/src/zeus-olimpic-1.0-2024-02-12.model"

# Output directory for generated files
OUTPUT_DIR = "/tmp/outputs"


class Predictor(BasePredictor):
    def setup(self):
        """Load the Zeus model into memory."""
        # Load model configuration
        with open(os.path.join(MODEL_DIR, "options.json"), "r") as f:
            self.options = json.load(f)

        # Load vocabulary (tags)
        self.tags = []
        with open(os.path.join(MODEL_DIR, "tags.txt"), "r") as f:
            for line in f:
                self.tags.append(line.rstrip("\r\n"))

        # Store key parameters
        self.height = self.options.get("height", 192)
        self.max_predict_length = self.options.get("max_predict_length", 700)
        self.rnn_dim = self.options.get("rnn_dim", 192)
        self.cnn_dim = self.options.get("cnn_dim", 32)
        self.cnn_stages = self.options.get("cnn_stages", 4)
        self.cnn_resblocks = self.options.get("cnn_resblocks", 2)
        self.rnn_layers = self.options.get("rnn_layers", 2)
        self.rnn_layers_decoder = self.options.get("rnn_layers_decoder", 1)
        self.timestep_width = self.options.get("timestep_width", 16)
        self.dropout = self.options.get("dropout", 0.2)

        # Build the model
        self.model = self._build_model()

        # Initialize by running a dummy inference (required for loading weights)
        dummy_input = tf.RaggedTensor.from_tensor(
            tf.ones([1, self.height, 128, 1], dtype=tf.float32),
            ragged_rank=2
        )
        self._decoder_inference(self.model["encoder"](dummy_input), 1)

        # Load weights
        self._load_weights(os.path.join(MODEL_DIR, "weights.h5"))

        print(f"Loaded Zeus model with {len(self.tags)} vocabulary tokens")

    def _build_model(self):
        """Build the encoder-decoder model architecture."""
        # Encoder
        inputs = tf.keras.layers.Input(
            shape=[self.height, None, 1], dtype=tf.float32, ragged=True
        )

        hidden = inputs.to_tensor()
        hidden = tf.keras.layers.Conv2D(self.cnn_dim, 3, 1, "same", use_bias=False)(hidden)

        for i in range(self.cnn_stages):
            filters = min(self.rnn_dim, self.cnn_dim * (1 << i))
            residual = tf.keras.layers.Conv2D(filters, 3, 2, "same", use_bias=False)(hidden)
            residual = tf.keras.layers.BatchNormalization()(residual)
            hidden = tf.keras.layers.Conv2D(filters, 3, 2, "same", use_bias=False)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            hidden = tf.keras.layers.Conv2D(filters, 3, 1, "same", use_bias=False)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden += residual
            hidden = tf.keras.layers.ReLU()(hidden)
            for _ in range(self.cnn_resblocks - 1):
                residual = hidden
                hidden = tf.keras.layers.Conv2D(filters, 3, 1, "same", use_bias=False)(hidden)
                hidden = tf.keras.layers.BatchNormalization()(hidden)
                hidden = tf.keras.layers.ReLU()(hidden)
                hidden = tf.keras.layers.Conv2D(filters, 3, 1, "same", use_bias=False)(hidden)
                hidden = tf.keras.layers.BatchNormalization()(hidden)
                hidden += residual
                hidden = tf.keras.layers.ReLU()(hidden)

        hidden = tf.transpose(hidden, [0, 2, 1, 3])
        hidden = tf.reshape(
            hidden,
            [tf.shape(hidden)[0], tf.shape(hidden)[1], hidden.shape[2] * hidden.shape[3]]
        )

        remaining = self.timestep_width // (1 << self.cnn_stages)
        if remaining > 1:
            hidden = tf.pad(hidden, [[0, 0], [0, (-tf.shape(hidden)[1]) % remaining], [0, 0]])
            hidden = tf.reshape(
                hidden,
                [tf.shape(hidden)[0], tf.shape(hidden)[1] // remaining, hidden.shape[2] * remaining]
            )

        hidden = tf.keras.layers.Dropout(self.dropout)(hidden)

        reduced_row_lengths = inputs.row_lengths(axis=2)[:, :1].to_tensor()[:, 0]
        reduced_row_lengths = (reduced_row_lengths + self.timestep_width - 1) // self.timestep_width
        mask = tf.sequence_mask(reduced_row_lengths)

        for layer in range(self.rnn_layers):
            residual = hidden
            rnn_layer = tf.keras.layers.LSTM(self.rnn_dim, return_sequences=True)
            hidden = tf.keras.layers.Bidirectional(rnn_layer, merge_mode="sum")(hidden, mask=mask)
            hidden = tf.keras.layers.Dropout(self.dropout)(hidden)
            if layer:
                hidden += residual

        encoder = tf.keras.Model(inputs=inputs, outputs=hidden)

        # Decoder layers
        target_embedding = tf.keras.layers.Embedding(1 + len(self.tags), self.rnn_dim)
        target_rnn = tf.keras.layers.RNN(
            WithAttention(
                [tf.keras.layers.LSTMCell(self.rnn_dim) for _ in range(self.rnn_layers_decoder)],
                self.rnn_dim
            ),
            return_sequences=True
        )
        target_output_layer = tf.keras.layers.Dense(1 + len(self.tags))

        return {
            "encoder": encoder,
            "target_embedding": target_embedding,
            "target_rnn": target_rnn,
            "target_output_layer": target_output_layer,
        }

    def _load_weights(self, weights_path: str):
        """Load weights from h5 file."""
        # Create a combined model for weight loading
        class CombinedModel(tf.keras.Model):
            def __init__(self, components):
                super().__init__()
                self.encoder = components["encoder"]
                self._target_embedding = components["target_embedding"]
                self._target_rnn = components["target_rnn"]
                self._target_output_layer = components["target_output_layer"]

        combined = CombinedModel(self.model)
        combined.built = True
        combined.load_weights(weights_path)

        # Update our model references
        self.model["encoder"] = combined.encoder
        self.model["target_embedding"] = combined._target_embedding
        self.model["target_rnn"] = combined._target_rnn
        self.model["target_output_layer"] = combined._target_output_layer

    @tf.function
    def _decoder_inference(self, encoded: tf.Tensor, max_length: tf.Tensor) -> tf.Tensor:
        """Run autoregressive decoding."""
        BOS = EOS = 0

        self.model["target_rnn"].cell.setup_memory(encoded)

        batch_size = tf.shape(encoded)[0]
        index = tf.zeros([], tf.int32)
        inputs = tf.fill([batch_size], BOS)
        states = self.model["target_rnn"].cell.get_initial_state(
            batch_size=batch_size, dtype=tf.float32
        )
        results = tf.TensorArray(tf.int32, size=max_length)
        result_lengths = tf.fill([batch_size], max_length)

        while tf.math.logical_and(
            index < max_length,
            tf.math.reduce_any(result_lengths == max_length)
        ):
            hidden = self.model["target_embedding"](inputs)
            hidden, states = self.model["target_rnn"].cell(hidden, states)
            hidden = self.model["target_output_layer"](hidden)
            predictions = tf.argmax(hidden, axis=-1, output_type=tf.int32)
            results = results.write(index, predictions)
            result_lengths = tf.where(
                (predictions == EOS) & (result_lengths > index),
                index,
                result_lengths
            )
            inputs = predictions
            index += 1

        results = tf.RaggedTensor.from_tensor(
            tf.transpose(results.stack()),
            lengths=result_lengths
        )
        return results

    def _preprocess_image(self, image_path: str) -> tf.Tensor:
        """Load and preprocess an image for inference."""
        # Read image file
        image_bytes = tf.io.read_file(image_path)
        image = tf.image.decode_image(image_bytes, channels=1, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Resize preserving aspect ratio
        image = tf.image.resize(
            image,
            size=[self.height, tf.int32.max],
            preserve_aspect_ratio=True,
            antialias=True
        )

        # Convert to ragged tensor format
        image = tf.RaggedTensor.from_tensor(
            tf.expand_dims(image, 0),  # Add batch dimension
            ragged_rank=2
        )

        return image

    def predict(
        self,
        image: CogPath = Input(description="Sheet music image (PNG, JPG, etc.)"),
        output_format: str = Input(
            description="Output format: 'lmx' for Linearized MusicXML tokens, 'musicxml' for standard MusicXML",
            default="musicxml",
            choices=["lmx", "musicxml"],
        ),
        max_length: int = Input(
            description="Maximum output sequence length",
            default=700,
            ge=100,
            le=2000,
        ),
    ) -> CogPath:
        """
        Recognize sheet music from an image and return musical notation.

        Output formats:
        - 'musicxml': Standard MusicXML file (.musicxml) that can be opened in notation
          software like MuseScore, Finale, or Sibelius.
        - 'lmx': Linearized MusicXML tokens file (.lmx).
        """
        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Preprocess image
        image_tensor = self._preprocess_image(str(image))

        # Run encoder
        encoded = self.model["encoder"](image_tensor, training=False)

        # Run decoder
        predictions = self._decoder_inference(encoded, max_length)
        predictions = predictions - 1  # Adjust for BOS offset

        # Convert token indices to strings
        tokens = []
        for idx in predictions[0].numpy():
            if 0 <= idx < len(self.tags):
                tokens.append(self.tags[idx])

        lmx_output = " ".join(tokens)

        if output_format == "lmx":
            output_path = os.path.join(OUTPUT_DIR, "output.lmx")
            with open(output_path, "w") as f:
                f.write(lmx_output)
            return CogPath(output_path)

        # Convert LMX to MusicXML
        delinearizer = Delinearizer()
        part_element = delinearizer.process_text(lmx_output)
        score_tree = part_to_score(part_element)

        # Write MusicXML file
        output_path = os.path.join(OUTPUT_DIR, "output.musicxml")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n')
            xml_string = ET.tostring(score_tree.getroot(), encoding="unicode")
            f.write(xml_string)

        return CogPath(output_path)


class WithAttention(tf.keras.layers.AbstractRNNCell):
    """Bahdanau attention cell for the decoder."""

    def __init__(self, cells, attention_dim):
        super().__init__()
        self._cells = cells
        self._project_encoder_layer = tf.keras.layers.Dense(attention_dim)
        self._project_decoder_layer = tf.keras.layers.Dense(attention_dim)
        self._output_layer = tf.keras.layers.Dense(1)

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    def setup_memory(self, encoded):
        self._encoded = encoded
        self._encoded_projected = self._project_encoder_layer(encoded)

    def call(self, inputs, states):
        projected = self._encoded_projected + tf.expand_dims(
            self._project_decoder_layer(tf.concat(states[0], axis=1)), axis=1
        )
        weights = tf.nn.softmax(self._output_layer(tf.tanh(projected)), axis=1)
        attention = tf.reduce_sum(self._encoded * weights, axis=1)
        inputs, new_states = tf.concat([inputs, attention], axis=1), []
        for i, (cell, state) in enumerate(zip(self._cells, states)):
            outputs, new_state = cell(inputs, state)
            inputs = outputs if i == 0 else inputs + outputs
            new_states.append(new_state)
        return outputs, tuple(new_states)
