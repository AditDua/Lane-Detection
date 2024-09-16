import numpy as np
#import cupy as np
import time
import torch
import cv2

def numpy_mul(matrix_size):
    A = np.full((matrix_size, matrix_size), 0.0, dtype='float')
    v1 = np.full(matrix_size, 0.0, dtype='float')
    v2 = np.full(matrix_size, 0.0, dtype='float')
	
    for i in range(matrix_size):
        for j in range(matrix_size):
            A[i, j] = float(i * matrix_size + j)


    for i in range(matrix_size):
        v1[i] = float(i)

    t1 = time.time()

    #for i in range(1):
    #v2 = np.matmul(A, v1)

    t2 = time.time()
    t_diff = t2 - t1
    print("TIme taken: ", t_diff)

    #print(A)
    #print(v1)
    #print(v2)

    #for (int i = 0; i < matrix_size; ++i)
        #printf("%.2f\n", v2[i])

def torch_mul(matrix_size):
    a = torch.rand(matrix_size, matrix_size)
    b = torch.rand(matrix_size, matrix_size)

    t1 = time.time()

    for i in range(1000):
        torch.matmul(a, b) # torch.Size([m, j])

    t2 = time.time()

    t_diff = t2 - t1
    print("time taken", t_diff)

def opencv_hls_select(img, sthresh=(0, 255),lthresh=()):
    # 1) Convert to HLS color space
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    L = hls_img[:,:,1]
    S = hls_img[:,:,2]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(S)
    binary_output[(S >= sthresh[0]) & (S <= sthresh[1])
                 & (L > lthresh[0]) & (L <= lthresh[1])] = 255
    
    return binary_output

def opencv_cuda_hls_select(img, sthresh=(0, 255),lthresh=()):
    # 1) Convert to HLS color space
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(img)
    gpu_hls_img = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_RGB2HLS)
    gpu_hls_img.download(img)
    # 2) Apply a threshold to the S channel
    L = cv2.cuda_GpuMat(gpu_hls_img.size(), cv2.CV_32FC1)
    S = cv2.cuda_GpuMat(gpu_hls_img.size(), cv2.CV_32FC1)
    cv2.cuda.split(gpu_hls_img, [L, S])
    #L = img[:,:,1]
    #S = img[:,:,2]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(S)
    binary_output[(S >= sthresh[0]) & (S <= sthresh[1])
                 & (L > lthresh[0]) & (L <= lthresh[1])] = 255  
    
    return binary_output

def main():
    #matrix_size = 1024*16
    #torch_mul(matrix_size)

    timers = {
        "full-pipeline": [],
    }

    cap = cv2.VideoCapture('videos/project_video.mp4')
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    while True:
        has, frame = cap.read()
        if has:
            t1 = time.time()
            frame = opencv_cuda_hls_select(frame, (140, 255), (120, 255))
            t2 = time.time()
            timers['full-pipeline'].append(t2 - t1)

            cv2.imshow('Frame', frame)
        else:
            break

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    of_fps = (num_frames - 1) / sum(timers["full-pipeline"])
    print("full Pipeline FPS : {:0.3f}".format(of_fps))

if __name__ == '__main__':
    main()