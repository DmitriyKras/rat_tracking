import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
from pycuda import autoinit
import cv2
from typing import Union


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class YOLOTRT:
    def __init__(self, model: str, input_shape: tuple,
                 conf=0.25, iou=0.45):
        """Class for YOLOv8 TensorRT inference

        Parameters
        ----------
        model : str
            Path to engine file
        input_shape : tuple
            Model's input shape (width, height)
        conf : float
            Confidence threshold, by default 0.25
        iou : float
            IoU threshold for NMS algo, by default 0.45
        """
        self.input_shape = input_shape
        self.dtype = np.float32
        self.conf = conf
        self.iou = iou
        self.logger = trt.Logger(trt.ILogger.INTERNAL_ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, model)
        self.max_batch_size = 1
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        """Load engine trt file
        """
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        """Allocate memory for inputs and outputs
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream

    def  __nms(self, boxes: np.ndarray) -> Union[np.ndarray, tuple]:
        """Perform postprocess and NMS on output boxes

        Parameters
        ----------
        boxes : np.ndarray
            All boxes from model output

        Returns
        -------
        np.ndarray or None
            Filtered boxes
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        conf = boxes[:,4]
        areas = w * h  # compute areas of boxes
        ordered = conf.argsort()[::-1]  # get sorted indexes of scores in descending order
        keep = []  # boxes to keep
        classes = np.zeros_like(conf)
        while ordered.size > 0:
            # Index of the current element:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[ordered[1:]])
            yy1 = np.maximum(y[i], y[ordered[1:]])
            xx2 = np.minimum(x[i] + w[i], x[ordered[1:]] + w[ordered[1:]])
            yy2 = np.minimum(y[i] + h[i], y[ordered[1:]] + h[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)
            iou = intersection / union
            indexes = np.where(iou <= self.iou)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        if len(keep) == 0:
            return ()
        boxes = np.concatenate((boxes[keep], conf[keep][..., None], classes[keep][..., None].reshape(-1, 1)), axis=1)
        return boxes

    def __call__(self, x: np.ndarray) -> Union[np.ndarray, tuple]:
        """Perform detection, NMS and boxes scaling

        Parameters
        ----------
        x : np.ndarray
            Plain input frame in BGR (H, W, C) np.uint8

        Returns
        -------
        Union[np.ndarray, tuple]
            Boxes scaled according to input frame size or empty tuple
        """
        # preprocess
        ratio = (x.shape[1] / self.input_shape[0], x.shape[0] / self.input_shape[1])
        ratio = np.array(ratio)
        x = cv2.resize(x, self.input_shape)
        x = x[:, :, ::-1].transpose(2, 0, 1)[None, ...] / 255.0
        x = x.astype(self.dtype)
        np.copyto(self.inputs[0].host, x.ravel())
        # transfer data on cuda device
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        # run computations
        self.context.execute_async(batch_size=1, bindings=self.bindings, stream_handle=self.stream.handle)
        # get data from cuda device
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
        self.stream.synchronize()
        boxes = self.outputs[0].host.reshape(5, -1).T
        # filter boxes by confidence
        f = boxes[:, 4] > self.conf
        boxes = boxes[f]
        boxes = self.__nms(boxes)  # nms boxes
        if len(boxes) > 0:
            boxes[:, :2] = boxes[:, :2] * ratio
            boxes[:, 2:4] = boxes[:, 2:4] * ratio
        return boxes
    
    def draw_bboxes(self, frame: np.ndarray, boxes: np.ndarray, xywh=True, color=(0, 0, 255)) -> np.ndarray:
        """Draw red bounding boxes on given frame
        
        Parameters
        ----------
        frame : np.ndarray
            Input frame
        boxes : np.ndarray
            Bounding boxes from model detection

        Returns
        -------
        np.ndarray
            Frame with boundung boxes
        """
        if len(boxes) > 0:
            if xywh:
                boxes[:, :2] = boxes[:, :2] - boxes[:, 2:4] / 2
                boxes[:, 2:4] = boxes[:, :2] + boxes[:, 2:4]
            for box in boxes:
                frame = cv2.rectangle(frame, (int(box[0]), int(box[1])),
                                      (int(box[2]), int(box[3])), color, 2)
                frame = cv2.putText(frame, 'Class {}: {:.2f}'.format(int(box[5]), box[4]), (int(box[0]), int(box[1])),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        return frame
    

class YOLOPoseTRT(YOLOTRT):
    def __init__(self, engine_path: str, input_shape: tuple,
                 conf=0.25, iou=0.45, n_kpts=17):
        """Class for YOLOv8-pose TensorRT inference

        Parameters
        ----------
        engine_path : str
            Path of engine file
        input_shape : tuple
            Model's input shape (width, height)
        conf : float
            Confidence YOLO threshold, by default 0.25
        iou : float
            IoU threshold for NMS algo, by default 0.45
        """
        super().__init__(engine_path, input_shape, conf, iou)
        self.n_kpts = n_kpts

    def  __nms(self, boxes: np.ndarray) -> Union[np.ndarray, tuple]:
        """Perform postprocess and NMS on output boxes

        Parameters
        ----------
        boxes : np.ndarray
            All boxes from YOLO output

        Returns
        -------
        np.ndarray or None
            Filtered boxes
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        conf = boxes[:, 4]
        areas = w * h  # compute areas of boxes
        ordered = conf.argsort()[::-1]  # get sorted indexes of scores in descending order
        keep = []  # boxes to keep
        while ordered.size > 0:
            # Index of the current element:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[ordered[1:]])
            yy1 = np.maximum(y[i], y[ordered[1:]])
            xx2 = np.minimum(x[i] + w[i], x[ordered[1:]] + w[ordered[1:]])
            yy2 = np.minimum(y[i] + h[i], y[ordered[1:]] + h[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)
            iou = intersection / union
            indexes = np.where(iou <= self.iou)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        if len(keep) == 0:
            return ()
        boxes = boxes[keep]
        return boxes
    
    def __call__(self, x: np.ndarray) -> Union[np.ndarray, tuple]:
        # preprocess
        ratio = (x.shape[1] / self.input_shape[0], x.shape[0] / self.input_shape[1])
        ratio = np.array(ratio)
        x = cv2.resize(x, self.input_shape)
        x = x[:, :, ::-1].transpose(2, 0, 1)[None, ...] / 255.0
        x = x.astype(self.dtype)
        np.copyto(self.inputs[0].host, x.ravel())
        # transfer data on cuda device
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        # run computations
        self.context.execute_async(batch_size=1, bindings=self.bindings, stream_handle=self.stream.handle)
        # get data from cuda device
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
        self.stream.synchronize()
        boxes = self.outputs[0].host.reshape(self.n_kpts * 3 + 5, -1).T
        boxes = boxes[boxes[:, 4] > self.conf, :]  # filter boxes by confidence
        boxes = self.__nms(boxes)  # nms boxes
        if len(boxes):
            out = np.empty((boxes.shape[0], 0))
            for i in range(self.n_kpts):
                boxes[:, 3*i + 5 : 3*i + 7] = boxes[:, 3*i + 5 : 3*i + 7] * ratio
                out = np.concatenate((out, boxes[:, 3*i + 5 : 3*i + 7]), axis=1)
            return out
        else:
            return ()
 