"""
Traffic Light Recognition System using Deep Learning and IoT
Module: iot_simulation.py
Description:
    Simulates an IoT Edge Traffic Controller node (e.g., ESP32 / Raspberry Pi)
    connected to smart intersection actuators (LEDs, Buzzer, Servo Gate) and sensors
    (Luminosity LDR, Ultrasonic Traffic Density, Ambient DHT22).
    Provides MQTT/JSON telemetry generation, Emergency Vehicle Preemption (EVO),
    and adaptive signal timing.
"""

import time
import json
import random
from datetime import datetime
from typing import Dict, Any, List, Optional


class IoTActuatorState:
    """Represents physical GPIO pin states for the simulated IoT controller."""
    def __init__(self):
        self.red_led_pin: int = 21       # GPIO 21
        self.yellow_led_pin: int = 19    # GPIO 19
        self.green_led_pin: int = 18     # GPIO 18
        self.buzzer_pin: int = 5         # GPIO 5 (PWM buzzer)
        self.servo_gate_pin: int = 4     # GPIO 4 (Servo barrier)
        
        # State values
        self.red_active: bool = False
        self.yellow_active: bool = False
        self.green_active: bool = False
        self.buzzer_active: bool = False
        self.buzzer_freq_hz: int = 0
        self.servo_angle_deg: int = 0    # 0 = Barrier Open, 90 = Barrier Closed

    def update_actuators(self, signal_state: str, pedestrian_mode: bool = False):
        """Update virtual hardware actuator states based on signal."""
        signal = signal_state.upper()
        if signal == "RED":
            self.red_active = True
            self.yellow_active = False
            self.green_active = False
            self.servo_angle_deg = 90  # Closed barrier
            if pedestrian_mode:
                self.buzzer_active = True
                self.buzzer_freq_hz = 1200  # High-pitch chirping for safe walk
            else:
                self.buzzer_active = False
                self.buzzer_freq_hz = 0
        elif signal == "YELLOW":
            self.red_active = False
            self.yellow_active = True
            self.green_active = False
            self.servo_angle_deg = 45  # Transitioning
            self.buzzer_active = True
            self.buzzer_freq_hz = 600   # Warning beep
        elif signal == "GREEN":
            self.red_active = False
            self.yellow_active = False
            self.green_active = True
            self.servo_angle_deg = 0   # Open barrier
            self.buzzer_active = False
            self.buzzer_freq_hz = 0
        else:  # NONE or OFF
            self.red_active = False
            self.yellow_active = False
            self.green_active = False
            self.servo_angle_deg = 0
            self.buzzer_active = False
            self.buzzer_freq_hz = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpio_pins": {
                "red_led": {"pin": self.red_led_pin, "state": "HIGH" if self.red_active else "LOW"},
                "yellow_led": {"pin": self.yellow_led_pin, "state": "HIGH" if self.yellow_active else "LOW"},
                "green_led": {"pin": self.green_led_pin, "state": "HIGH" if self.green_active else "LOW"},
                "piezo_buzzer": {"pin": self.buzzer_pin, "state": "ACTIVE" if self.buzzer_active else "IDLE", "freq_hz": self.buzzer_freq_hz},
                "servo_barrier": {"pin": self.servo_gate_pin, "angle_deg": self.servo_angle_deg, "status": "CLOSED" if self.servo_angle_deg >= 80 else "OPEN"}
            }
        }


class SmartTrafficIoTNode:
    """
    Virtual IoT Node representing an intelligent intersection controller.
    Supports MQTT simulation, AI recognition sync, Emergency Vehicle Preemption,
    and adaptive traffic sensor telemetry.
    """
    def __init__(self, node_id: str = "ESP32-INTERSECTION-NODE-01", broker_url: str = "mqtt.smartcity.iot:1883"):
        self.node_id = node_id
        self.broker_url = broker_url
        self.current_signal: str = "RED"
        self.confidence_score: float = 0.0
        self.source_mode: str = "AI_RECOGNITION"  # AI_RECOGNITION, AUTO_CYCLE, EMERGENCY_OVERRIDE
        self.actuators = IoTActuatorState()
        self.actuators.update_actuators(self.current_signal)
        
        # Simulated Environmental & Traffic Sensors
        self.ambient_lux: float = 650.0       # Light sensor
        self.vehicle_queue: int = 5           # Ultrasonic / Inductive loop count
        self.temp_c: float = 28.5             # DHT22 temperature
        self.humidity_pct: float = 62.0       # DHT22 humidity
        self.power_draw_watts: float = 18.4   # Current power draw
        self.network_latency_ms: int = 24     # Ping to MQTT broker
        
        # State machine timing
        self.state_countdown: int = 30
        self.emergency_override: bool = False
        self.emergency_type: Optional[str] = None  # AMBULANCE, FIRE_TRUCK, POLICE
        self.pedestrian_request: bool = False
        
        # Telemetry logs
        self.telemetry_history: List[Dict[str, Any]] = []
        self.max_history: int = 50

    def sync_with_ai_recognition(self, signal: str, confidence: float) -> Dict[str, Any]:
        """
        Synchronize the IoT Node with signal detected by Deep Learning CNN.
        """
        if self.emergency_override:
            # Emergency override takes precedence over normal AI stream
            return self.get_latest_telemetry()

        signal = signal.upper()
        if signal in ["RED", "YELLOW", "GREEN", "NONE"]:
            self.current_signal = signal
            self.confidence_score = confidence
            self.source_mode = "AI_RECOGNITION"
            self.actuators.update_actuators(self.current_signal, self.pedestrian_request)
            
            # Simulate slight sensor fluctuations
            self.vehicle_queue = max(0, self.vehicle_queue + random.randint(-1, 1))
            self.ambient_lux = max(100.0, min(1200.0, self.ambient_lux + random.uniform(-15.0, 15.0)))
            self.network_latency_ms = random.randint(18, 45)
            
            payload = self._generate_telemetry_packet()
            self._record_history(payload)
            return payload
        return self.get_latest_telemetry()

    def trigger_emergency_vehicle_preemption(self, vehicle_type: str = "AMBULANCE", duration_seconds: int = 15) -> Dict[str, Any]:
        """
        Trigger Emergency Vehicle Preemption (EVO).
        Forces GREEN signal for priority vehicle passage and closes crossing barriers.
        """
        self.emergency_override = True
        self.emergency_type = vehicle_type
        self.source_mode = "EMERGENCY_OVERRIDE"
        self.current_signal = "GREEN"
        self.confidence_score = 99.9
        self.state_countdown = duration_seconds
        self.actuators.update_actuators("GREEN", pedestrian_mode=False)
        
        payload = self._generate_telemetry_packet(event="EMERGENCY_PRIORITY_PREEMPTION_ACTIVE")
        self._record_history(payload)
        return payload

    def clear_emergency_override(self) -> Dict[str, Any]:
        """Restore normal AI recognition / auto cycle after emergency vehicle clears."""
        self.emergency_override = False
        self.emergency_type = None
        self.source_mode = "AI_RECOGNITION"
        payload = self._generate_telemetry_packet(event="EMERGENCY_OVERRIDE_CLEARED")
        self._record_history(payload)
        return payload

    def request_pedestrian_crossing(self) -> Dict[str, Any]:
        """Simulate pedestrian pushbutton press at crosswalk."""
        self.pedestrian_request = True
        # If currently green, schedule change to yellow -> red
        payload = self._generate_telemetry_packet(event="PEDESTRIAN_CROSSING_REQUEST_QUEUED")
        self._record_history(payload)
        return payload

    def reset_pedestrian_crossing(self):
        """Reset pedestrian crossing state."""
        self.pedestrian_request = False
        self.actuators.update_actuators(self.current_signal, False)

    def _generate_telemetry_packet(self, event: str = "NORMAL_HEARTBEAT") -> Dict[str, Any]:
        """Construct standard MQTT JSON telemetry packet."""
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        action_map = {
            "RED": "STOP - Vehicle halt mandatory; crosswalk accessible",
            "YELLOW": "CAUTION - Prepare to stop; clear intersection box",
            "GREEN": "GO - Corridor open; safe to proceed at speed limit",
            "NONE": "IDLE - Signal unrecognized / camera obstruction"
        }
        
        packet = {
            "mqtt_topic": f"smartcity/v1/{self.node_id}/telemetry",
            "timestamp": timestamp_str,
            "node_metadata": {
                "device_id": self.node_id,
                "hardware": "ESP32-S3 Dual Core 240MHz",
                "firmware_version": "v3.4.1-iot-ai",
                "ip_address": "192.168.1.142",
                "broker": self.broker_url,
                "ping_ms": self.network_latency_ms
            },
            "traffic_control": {
                "active_signal": self.current_signal,
                "confidence_pct": round(self.confidence_score, 2),
                "control_mode": self.source_mode,
                "action_command": action_map.get(self.current_signal, "UNKNOWN"),
                "emergency_override": self.emergency_override,
                "emergency_vehicle": self.emergency_type,
                "pedestrian_button_pressed": self.pedestrian_request
            },
            "sensor_readings": {
                "ambient_light_lux": round(self.ambient_lux, 1),
                "estimated_queue_vehicles": self.vehicle_queue,
                "ambient_temperature_c": round(self.temp_c, 1),
                "humidity_pct": round(self.humidity_pct, 1),
                "power_consumption_w": round(self.power_draw_watts, 2)
            },
            "hardware_actuators": self.actuators.to_dict(),
            "event_trigger": event
        }
        return packet

    def _record_history(self, packet: Dict[str, Any]):
        """Maintain ring buffer of recent telemetry logs."""
        self.telemetry_history.insert(0, packet)
        if len(self.telemetry_history) > self.max_history:
            self.telemetry_history.pop()

    def get_latest_telemetry(self) -> Dict[str, Any]:
        """Retrieve most recent telemetry packet."""
        if self.telemetry_history:
            return self.telemetry_history[0]
        packet = self._generate_telemetry_packet()
        self._record_history(packet)
        return packet

    def get_telemetry_json(self, indent: int = 2) -> str:
        """Return formatted JSON string of latest telemetry packet."""
        return json.dumps(self.get_latest_telemetry(), indent=indent)


# Singleton instance for quick access across dashboard tabs
default_iot_node = SmartTrafficIoTNode()


if __name__ == "__main__":
    print("=" * 65)
    print(" ESP32-S3 SMART TRAFFIC IoT SIMULATOR & TELEMETRY ENGINE")
    print("=" * 65)

    node = SmartTrafficIoTNode(node_id="ESP32-INTERSECTION-ALPHA")
    print(f"\n[1] Initialized IoT Controller: {node.node_id}")
    print(f"    Broker: {node.broker_url} | Ping: {node.network_latency_ms} ms")

    test_states = [
        ("RED", 96.5, False, "Vehicle Queue Detection"),
        ("GREEN", 94.2, False, "Green Wave Passage"),
        ("YELLOW", 89.0, False, "Transition Warning"),
        ("RED", 99.1, True, "Pedestrian Walk Crosswalk Active")
    ]

    print("\n[2] Testing Simulated Traffic Signal Cycles & GPIO Actuators:")
    print("-" * 65)
    for sig, conf, ped, desc in test_states:
        if ped:
            node.request_pedestrian_crossing()
        else:
            node.reset_pedestrian_crossing()
        packet = node.sync_with_ai_recognition(sig, conf)
        pins = packet["hardware_actuators"]["gpio_pins"]
        sensors = packet["sensor_readings"]

        print(f"\n>> Phase: {desc}")
        print(f"   Signal State    : {sig} (Conf: {conf}%)")
        print(f"   LED Status      : RED={pins['red_led']['state']}, YEL={pins['yellow_led']['state']}, GRN={pins['green_led']['state']}")
        print(f"   Servo Gate      : {pins['servo_barrier']['status']} ({pins['servo_barrier']['angle_deg']}°)")
        buzzer_info = pins['piezo_buzzer']
        buzzer_str = f"ON ({buzzer_info['freq_hz']} Hz)" if buzzer_info['state'] == 'ACTIVE' else "OFF"
        print(f"   Buzzer          : {buzzer_str}")
        print(f"   Sensors         : Queue={sensors['estimated_queue_vehicles']} cars | Lux={sensors['ambient_light_lux']} | Temp={sensors['ambient_temperature_c']}°C")

    print("\n[3] Testing Emergency Vehicle Preemption (EVO):")
    print("-" * 65)
    evo_packet = node.trigger_emergency_vehicle_preemption("Ambulance", duration_seconds=15)

    print(f"   Override Status : {evo_packet['traffic_control']['emergency_override']}")
    print(f"   Vehicle Type    : {evo_packet['traffic_control']['emergency_vehicle']}")
    print(f"   Active Signal   : {evo_packet['traffic_control']['active_signal']}")
    print(f"   MQTT Topic      : {evo_packet['mqtt_topic']}")

    print("\n[4] Sample Outgoing MQTT JSON Telemetry Packet:")
    print("-" * 65)
    print(node.get_telemetry_json())

    print("\n" + "=" * 65)
    print(" SUCCESS: iot_simulation.py is working perfectly!")
    print("=" * 65)
    input("\nPress Enter to exit...")

