from config import ANATOMICAL_LANDMARKS, CVM_STAGES
import time
import json
import os


def format_landmarks(
    ceph_id: str,
    landmarks: list,
) -> dict:
    
    data = dict()

    data["ceph_id"] = ceph_id
    
    data["landmarks"] = []
    for index, landmark in enumerate(ANATOMICAL_LANDMARKS.items()):
        landmark_id = landmark[0]
        info = landmark[1]

        data["landmarks"].append(
            {
                "landmark_id": landmark_id,
                "title": info["title"],
                "symbol": info["symbol"],
                "value": {
                    "x": round(landmarks[index][0]),
                    "y": round(landmarks[index][1])
                }
            }
        )
    data["dataset_name"] = "CEPHA29: Cephalometric Landmark Detection Dataset"
    data["dataset_split"] = "Validation"
    data["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S.000000")

    return data


def format_cvm_stage(
    ceph_id: str,
    cvm_stage: int
) -> dict:
    data = dict()

    data["ceph_id"] = ceph_id
    key = [key for key, value in CVM_STAGES.items() if value["value"] == cvm_stage][0]
    
    data["cvm_stage"] = {
        "id": key,
        "title": CVM_STAGES[key]["title"],
        "value": CVM_STAGES[key]["value"]
    }
    data["dataset_name"] = "CEPHA29: Cephalometric Landmark Detection Dataset"
    data["dataset_split"] = "Validation"
    data["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S.000000")

    return data
    

def save_landmarks_results(
    folder_path: str,
    ceph_id: str,
    landmarks: list
) -> None:
    """
    Saves the predicted values of landmarks in .json format.
    Args:
        folder_path: Relative path of the directory where to save the results.
        e.g. `./cephalometric-landmarks-detection/results/valid/Cephalometric Landmarks`
        ceph_id: Name/ID of the cephalogram without extension.
        e.g. `cks2ip8fq2a0o0yufhfuo7clg`
        landmarks: List/Numpy array of shape (29, 2) of predicted landmarks.
    """
    
    data = format_landmarks(ceph_id, landmarks)
    
    file_name = ceph_id + "." + "json"
    with open(os.path.join(folder_path, file_name), "w") as file:
        json.dump(data, file)

        
def save_cvm_results(
    folder_path: str,
    ceph_id: str,
    cvm_stage: int
) -> None:
    """
    Saves the predicted values of landmarks in .json format.
    Args:
        folder_path: Relative path of the directory where to save the results.
        e.g. `./cephalometric-landmarks-detection/results/valid/CVM Stages`
        ceph_id: Name/ID of the cephalogram without extension.
        e.g. `cks2ip8fq2a0o0yufhfuo7clg`
        cvm_stage: Predicted CVM Stage.
        e.g. `1 or 2 or 5`
    """
    
    data = format_cvm_stage(ceph_id, cvm_stage)
    
    file_name = ceph_id + "." + "json"
    with open(os.path.join(folder_path, file_name), "w") as file:
        json.dump(data, file)


if __name__ == "__main__":
    import numpy as np
    import os

    landmarks = np.random.randint(0, 2000, size=(29, 2)) # Your Model's Predicted Landmarks
    cvm_stage = np.random.randint(0, 5) # Your Model's predicted CVM Stage

    save_landmarks_results(folder_path="./results/valid/Cephalometric Landmarks", ceph_id="cks2ip8fq2a0o0yufhfuo7clg", landmarks=landmarks)
    save_cvm_results(folder_path="./results/valid/CVM Stages", ceph_id="cks2ip8fq2a0o0yufhfuo7clg", cvm_stage=cvm_stage)
