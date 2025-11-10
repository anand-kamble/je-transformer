from __future__ import annotations

import argparse
import os

# Ensure KERAS_BACKEND default
if not os.environ.get("KERAS_BACKEND"):
	os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf

from models.je_model import build_je_model


def main():
	parser = argparse.ArgumentParser(description="Export JE pointer model as a TensorFlow SavedModel.")
	parser.add_argument("--output-dir", type=str, required=True, help="Local path or gs:// URI to save the SavedModel")
	parser.add_argument("--encoder", type=str, default="bert-base-multilingual-cased")
	parser.add_argument("--hidden-dim", type=int, default=256)
	parser.add_argument("--max-lines", type=int, default=8)
	parser.add_argument("--temperature", type=float, default=1.0)
	args = parser.parse_args()

	model = build_je_model(
		encoder_loc=args.encoder,
		hidden_dim=args.hidden_dim,
		max_lines=args.max_lines,
		temperature=float(args.temperature),
	)

	# Export
	tf.saved_model.save(model, args.output-dir if not args.output_dir.startswith("gs://") else "/tmp/je_saved_model")

	# If gs:// path, upload directory recursively
	if args.output_dir.startswith("gs://"):
		from google.cloud import storage
		client = storage.Client()
		_, path = args.output_dir.split("gs://", 1)
		bucket_name, prefix = path.split("/", 1)
		bucket = client.bucket(bucket_name)
		for root, _, files in os.walk("/tmp/je_saved_model"):
			for f in files:
				local_path = os.path.join(root, f)
				rel = os.path.relpath(local_path, "/tmp/je_saved_model")
				blob = bucket.blob(f"{prefix.rstrip('/')}/{rel}")
				blob.upload_from_filename(local_path)
		print(f"SavedModel uploaded to {args.output_dir}")
	else:
		print(f"SavedModel written to {args.output_dir}")


if __name__ == "__main__":
	main()



