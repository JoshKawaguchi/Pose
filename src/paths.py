import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEQUENCES_PATH = os.path.join(BASE_DIR, "sequences", "*")
METADATA_PATH = os.path.join(BASE_DIR, "metadata", "metadata.json")
STATICPOSE_PATH = os.path.join(BASE_DIR, "metadata", "staticPose.mat")