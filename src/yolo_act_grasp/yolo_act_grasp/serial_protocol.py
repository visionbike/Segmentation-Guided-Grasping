import struct
from typing import NamedTuple, Sequence

class Protocol:
    """
    Teensy U2D2 serial wire-format constants.
    Packet layout (37 bytes; big-endian int16 angles in 0.1-deg units):
        [0]      header 0x3E
        [1]      message id   (0xC1 feedback / 0xB2 command)
        [2]      subtype      (0x02 feedback / 0x00 command)
        [3]      length 0x25  (== 37)
        [4:18]   right arm R1-R7  (7 x int16)
        [18:32]  left arm  L1-L7  (7 x int16)
        [32]     right gripper    (0/1)
        [33]     left gripper     (0/1)
        [34:36]  neck             (int16)
        [36]     checksum = sum(bytes[0:36]) & 0xFF
    """
    HEADER:             int = 0x3E
    FEEDBACK_ID:        int = 0xC1      # byte[1] for feedback packets
    COMMAND_ID:         int = 0xB2      # byte[1] for command packets
    FEEDBACK_SUBTYPE:   int = 0x02      # byte[2] for feedback packets
    COMMAND_SUBTYPE:    int = 0x00      # byte[2] for command packets
    PACKET_LENGTH:      int = 37        # value of the length (0x25) and the full packet size
    CHECKSUM_INDEX:     int = 36        # last value; checksum covers bytes [0:36)


def compute_checksum(buffer: Sequence[int]) -> int:
    """
    Compute the packet checksum.

    :param buffer: Packet bytes; only indices [0:36) are summed.
    :return: Sum of bytes [0:36) masked to one byte.
    """
    return sum(buffer[: Protocol.CHECKSUM_INDEX]) & 0xFF


def verify_checksum(packet: Sequence[int]) -> bool:
    """
    Check a packet's trailing checksum byte against its contents.

    :param packet: A complete 37-byte packet.
    :return: True if ``packet[36]`` equals the computed checksum.
    """
    return compute_checksum(packet) == packet[Protocol.CHECKSUM_INDEX]


def encode_joint_command(
    right_arm_deg: Sequence[float],
    left_arm_deg: Sequence[float],
    right_gripper_pred: float,
    left_gripper_pred: float,
    neck_deg: float
) -> bytes:
    """
    Encode a 37-byte command packet from degree values.
    Arm angles are packed as big-endian INT16 in 0.1-deg units; grippers are thresholded to 0/1.

    :param right_arm_deg: 7 right-arm joint angles in degrees.
    :param left_arm_deg: 7 left_arm joint angles in degrees.
    :param right_gripper_pred: right gripper predicted command.
    :param left_gripper_pred: left gripper predicted command.
    :param neck_deg: neck joint angle in degrees.
    :return: the 37-byte command packet, checksum included.
    """
    right_arm_deci_deg = [int(round(v * 10)) for v in right_arm_deg]
    left_arm_deci_deg = [int(round(v * 10)) for v in left_arm_deg]
    neck_deci_deg = int(round(neck_deg * 10))
    right_gripper_cmd = 1 if float(right_gripper_pred) > 0.5 else 0
    left_gripper_cmd = 1 if float(left_gripper_pred) > 0.5 else 0

    buffer = bytearray(Protocol.PACKET_LENGTH)
    buffer[0] = Protocol.HEADER
    buffer[1] = Protocol.COMMAND_ID
    buffer[2] = Protocol.COMMAND_SUBTYPE
    buffer[3] = Protocol.PACKET_LENGTH
    buffer[4: 18] = struct.pack(">7h", *right_arm_deci_deg)
    buffer[18: 32] = struct.pack(">7h", *left_arm_deci_deg)
    buffer[32] = right_gripper_cmd
    buffer[33] = left_gripper_cmd
    buffer[34: 36] = struct.pack(">h", neck_deci_deg)
    buffer[Protocol.CHECKSUM_INDEX] = compute_checksum(buffer)
    return bytes(buffer)


class JointState(NamedTuple):
    """
    Decoded 17-dim feedback frame (arm and neck angles in degrees).
    """
    right_arm_deg: tuple
    left_arm_deg: tuple
    right_gripper_status: int
    left_gripper_status: int
    neck_deg: float


def decode_joint_state(packet: Sequence[int]) -> JointState:
    """
    Decode verified 37-byte feedback packet into degree values.
    Assume the packet is complete and checksum-verified. The caller converts to radians as needed.

    :param packet: a complete 37-byte feedback packet (header 0x3E 0xC1 0x02 0x25).
    :return: a ``JointState`` with arm/neck angles in degrees and the two gripper statuses as 0/1.
    """
    right_arm_deci_deg = struct.unpack(">7h", bytes(packet[4: 18]))
    left_arm_deci_deg = struct.unpack(">7h", bytes(packet[18: 32]))
    neck_deci_deg = struct.unpack(">h", bytes(packet[34: 36]))[0]
    return JointState(
        right_arm_deg=tuple(v / 10.0 for v in right_arm_deci_deg),
        left_arm_deg=tuple(v / 10.0 for v in left_arm_deci_deg),
        right_gripper_status=packet[32],
        left_gripper_status=packet[33],
        neck_deg=neck_deci_deg / 10.0
    )


def clamp_joint_angles(values: Sequence[float], limits: Sequence[tuple]) -> list[float]:
    """
    Clamp each value into its (low, high) limit. The caller detects and logs out-of-range joints by comparing the
    result to the input.

    :param values: joint angles to clamp (any unit; typically degrees).
    :param limits: same-length sequence of ``(low, high)`` bounds, one per value.
    :return: the clamped values, in the same order as ``values``.
    """
    if len(values) != len(limits):
        raise ValueError(f"[ERROR] `values` length {len(values)} != `limits` length {len(limits)}.")
    return [max(low, min(high, v)) for v, (low, high) in zip(values, limits)]
