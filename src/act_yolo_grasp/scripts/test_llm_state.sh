#!/bin/bash
# Usage: ./test_llm_state.sh <state_name>
#
# /llm_state format: [t0, t1, t2, d0, d1, d2]
#   bits 0-2 = target      : red_cookies, green_tea, yellow_cookies
#   bits 3-5 = destination : mid_basket,  right_basket, human_hand
#
# Examples:
#   ./test_llm_state.sh red_cookies_to_mid_basket
#   ./test_llm_state.sh all_gray
#   ./test_llm_state.sh full_reset

STATE=${1:-""}

case "$STATE" in
  red_cookies_to_mid_basket)      DATA="[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]" ;;
  red_cookies_to_right_basket)    DATA="[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]" ;;
  green_tea_to_mid_basket)        DATA="[0.0, 1.0, 0.0, 1.0, 0.0, 0.0]" ;;
  green_tea_to_right_basket)      DATA="[0.0, 1.0, 0.0, 0.0, 1.0, 0.0]" ;;
  yellow_cookies_to_mid_basket)   DATA="[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]" ;;
  red_cookies_to_human_hand)      DATA="[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]" ;;
  green_tea_to_human_hand)        DATA="[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]" ;;
  yellow_cookies_to_human_hand)   DATA="[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]" ;;
  all_gray)                       DATA="[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]" ;;
  full_reset)                     DATA="[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]" ;;
  *)
    echo "Unknown state: '$STATE'"
    echo ""
    echo "Available states:"
    echo "  red_cookies_to_mid_basket    -> [1,0,0, 1,0,0]"
    echo "  red_cookies_to_right_basket  -> [1,0,0, 0,1,0]"
    echo "  green_tea_to_mid_basket      -> [0,1,0, 1,0,0]"
    echo "  green_tea_to_right_basket    -> [0,1,0, 0,1,0]"
    echo "  yellow_cookies_to_mid_basket -> [0,0,1, 1,0,0]"
    echo "  red_cookies_to_human_hand    -> [1,0,0, 0,0,1]"
    echo "  green_tea_to_human_hand      -> [0,1,0, 0,0,1]"
    echo "  yellow_cookies_to_human_hand -> [0,0,1, 0,0,1]"
    echo "  all_gray                     -> [0,0,0, 0,0,0]"
    echo "  full_reset                   -> [1,1,1, 1,1,1]"
    exit 1
    ;;
esac

echo "[TEST] state=$STATE  data=$DATA"
# yolo_seg_node subscribes /llm_state with RELIABLE + TRANSIENT_LOCAL QoS.
# Must match here or the node silently drops every message (incompatible DURABILITY).
ros2 topic pub /llm_state std_msgs/msg/Float32MultiArray "{data: $DATA}" -r 5 \
  --qos-reliability reliable --qos-durability transient_local
