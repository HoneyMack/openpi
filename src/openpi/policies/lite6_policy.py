import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from PIL import Image


def make_lite6_example() -> dict:
    """Creates a random input example for the Lite6 policy."""
    return {
        "wrist_rgb": np.random.randint(256, size=(640, 480, 3), dtype=np.uint8),
        "base_rgb": np.random.randint(256, size=(640, 480, 3), dtype=np.uint8),
        "state": np.ones((7,)), # 6 joints and 1 gripper
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class Lite6Inputs(transforms.DataTransformFn):
    """Inputs for the Lite6 policy.

    Expected inputs:

    - wrist_rgb:[H, W, 3]
    - base_rgb: [H, W, 3]
    - state: [7] # 6 joints and 1 gripper
    - actions: [action_horizon, 7] # Actions are only available during training.
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # If true, this will convert the joint and gripper values from the standard Lite6 space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        data = _decode_lite6(data, adapt_to_pi=self.adapt_to_pi)

        # Get the state. We are padding from 14 to the model action dim.
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": data["base_rgb"],
                "left_wrist_0_rgb": data["wrist_rgb"],
                "right_wrist_0_rgb": np.zeros_like(data["base_rgb"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Lite6Outputs(transforms.DataTransformFn):
    """Outputs for the Lite6 policy."""

    # If true, this will convert the joint and gripper values from the standard Lite6 space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims.
        actions = np.asarray(data["actions"][:, :7])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between lite6 and pi joint angles."""
    return np.array([1, -1, 1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Lite6 transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the lite6 OpenParallelGripper:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = (
        _unnormalize(value, min_val=0, max_val=0.032) + 0.01844
    )  # alohaではなぜか0.01844が最小値になっていたので，それに合わせる

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return _normalize(value, min_val=0.4, max_val=1.5)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by lite6 OpenParallelGripper.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = _unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the OpenParallelGripper code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=0.0, max_val=1.0)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=0.0, max_val=1.0)
    return _normalize(value, min_val=0.4, max_val=1.5)


def _decode_lite6(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [arm_joint_angles,gripper]
    # dim sizes: [6, 1]
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        img = einops.rearrange(img, "c h w -> h w c")

        size = (224, 224)  # pi0の画像サイズに合わせる
        img = Image.fromarray(img)
        img = img.resize(size, Image.Resampling.BICUBIC)
        return np.array(img)

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Lite6 runtime.
        state[6] = _gripper_to_angular(state[6])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, 6] = _gripper_from_angular(actions[:, 6])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, 6] = _gripper_from_angular_inv(actions[:, 6])
    return actions
