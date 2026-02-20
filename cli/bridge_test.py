"""Bridge smoke test -- quick validation of the 4-tool Bridge API."""
import json
import sys
from bridge import ROSBagBridge


def main():
    bridge = ROSBagBridge()
    if len(sys.argv) < 2:
        print("Usage: python -m cli.bridge_test <bag_path>")
        sys.exit(1)

    bag_path = sys.argv[1]

    print("\n=== BAG METADATA ===")
    meta = bridge.get_bag_metadata(bag_path)
    print(json.dumps(meta, indent=2)[:3000])

    print("\n=== TOPIC STATISTICS: /odom (first 30s) ===")
    stats = bridge.get_topic_statistics(bag_path, "/odom",
                                         start_time=meta["start_time"],
                                         end_time=meta["start_time"] + 30,
                                         window_size=10.0)
    print(json.dumps(stats, indent=2)[:3000])

    print("\n=== FREQUENCY CHECK: /odom ===")
    freq = bridge.check_topic_frequency(bag_path, "/odom", resolution=5.0)
    print(json.dumps({k: v for k, v in freq.items() if k != "frequency_series"}, indent=2))
    print(f"  (series has {len(freq.get('frequency_series', []))} entries)")

    print("\n=== SAMPLE MESSAGES: /chassis_cmd_vel ===")
    samples = bridge.sample_messages(bag_path, "/chassis_cmd_vel", count=3)
    print(json.dumps(samples, indent=2)[:2000])


if __name__ == "__main__":
    main()
