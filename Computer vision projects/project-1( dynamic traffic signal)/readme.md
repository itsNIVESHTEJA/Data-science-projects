Project Architecture: 
---------------------
1. **Input Layer**:
   - Four video streams simulate real-time traffic from each road connected to the junction.

2. **Detection Layer**:
   - YOLOv5 is used to detect vehicles in each frame.
   - Model returns labels and bounding boxes for each vehicle.

3. **Weight Calculation**:
   - Vehicles are assigned weights depending on type (bike, car, truck, etc.)
   - Total weight is computed for each direction.

4. **Signal Decision Module**:
   - The direction with the highest vehicle weight gets the green signal.
   - Others are given red or yellow signals based on timing.

5. **Output Layer**:
   - Combined 2x2 grid of all video streams.
   - Real-time display of bounding boxes, signal lights, and direction labels.

Modules Used:
-------------
- **OpenCV (cv2)**: Video streaming, display, frame handling.
- **Torch (PyTorch)**: Loading and running YOLOv5 model.
- **OS / Time**: File handling and signal timing logic.
- **NumPy**: Efficient frame manipulation.
- **YOLOv5**: For pre-trained object detection (via PyTorch Hub).

Customizable Parameters:
------------------------
- **Signal duration**: Max green time, yellow time.
- **Weight factors**: You can modify the weight given to each vehicle type.
- **Model**: Swap between `yolov5s`, `yolov5m`, or custom trained models.
- **Camera Inputs**: Use `cv2.VideoCapture(index)` for real camera feed.

Performance Notes:
------------------
- Designed to work efficiently on mid-range laptops.
- Processing is optimized by resizing frames before passing to YOLOv5.
- You can set `torch.device("cuda")` if GPU is available to improve performance.

Sample Use Case:
----------------
Imagine a busy intersection in a city like Nizamabad. This system can:
- Reduce idle time for low-traffic roads.
- Prioritize ambulance or VIP vehicle lanes (future enhancement).
- Collect traffic data for smart city planning.

Security & Privacy:
-------------------
This system runs locally and doesn't transmit any video to the cloud. It's safe to use in schools, campuses, or city surveillance without privacy concerns.

How to Contribute:
------------------
1. Fork this repository
2. Clone the forked repo to your system
3. Make your changes and push to your fork
4. Submit a Pull Request

We welcome:
- Code improvements
- Model performance boosts
- UI enhancements (like adding dashboards or analytics)
- Integration with IoT devices or Raspberry Pi


