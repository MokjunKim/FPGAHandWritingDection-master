#==================================================
#                import package
#=================================================
import math
from pynq import Overlay
import numpy as np
from PIL import Image as PIL_Image
from pynq import MMIO
from pynq import Xlnk
import time
import datetime
import ctypes
import cv2
import os
import queue
import re
import sys

#==================================================
#                Image Input Path
#=================================================
img_path = "image.png"

#==================================================
#                Load overlay  and IP
#=================================================
overlay = Overlay("./hw_bd/hmc.bit")

xlnk = Xlnk()   
xlnk.xlnk_reset()

#==================================================
#            allocated the memory  inbuff
#=================================================
weight_base_buffer = xlnk.cma_array(shape=(4260640,), dtype=np.int32)
print("Weight: 16M",weight_base_buffer.physical_address)
WEIGHT_BASE = weight_base_buffer.physical_address

beta_base_buffer = xlnk.cma_array(shape=(1134,), dtype=np.int32)
print("Bias: 4K",beta_base_buffer.physical_address)
BETA_BASE=beta_base_buffer.physical_address

img_base_buffer = xlnk.cma_array(shape=(13312,), dtype=np.int32)
print("Img: 52K",img_base_buffer.physical_address)
IMG_MEM = img_base_buffer.physical_address

#===============================================
#        weight and bais copyto memory
#==============================================
params_wight = np.fromfile("./parameters/weight.bin", dtype=np.int32)
np.copyto(weight_base_buffer, params_wight)
print("Weight copy ok")

params_bais = np.fromfile("./parameters/bias.bin", dtype=np.int32)
np.copyto(beta_base_buffer, params_bais)
print("Bias copy ok")

#===============================================
#                network data
#==============================================
weight_offset = [800, 51200, 4194304]
beta_offset = [32, 64, 1024]
M_value = [94, 48, 43]

K=5
Tn=4
Tm=32
Tr=32
Tc=32
ALPHA_BETA_MAX_NUM=1024

SIZE = 32

SYMBOL = {0: '0',
          1: '1',
          2: '2',
          3: '3',
          4: '4',
          5: '5',
          6: '6',
          7: '7',
          8: '8',
          9: '9',
          10: '+',
          11: '-',
          12: '*',
          13: '/'}

SCALE = 0.007874015748031496

MEM_BASE = IMG_MEM
MEM_LEN = 16*16*32*4 + 8*8*64*4
Memory_top = MEM_BASE
Memory_bottom = MEM_BASE + MEM_LEN

in_ptr  = np.zeros(4)
out_ptr = np.zeros(4)

in_ptr[0] = Memory_top
out_ptr[0] = Memory_bottom - 16*16*32*4

in_ptr[1] = out_ptr[0]
out_ptr[1] = Memory_top

in_ptr[2] = out_ptr[1]
out_ptr[2] = Memory_bottom - 1024*4

in_ptr[3] = out_ptr[2]
out_ptr[3] = Memory_top

#===============================================
#              image pre-processing
#==============================================
def quantized_np(array,scale,data_width=8):
    """Quantify the input data of network,
    Args:
        array: input image data array. In this project its shape is [1,32*32].
        scale: Quantized parameter. Calculate offline.
        data_width: Quantized parameter. Define the quantization accuracy.
    Returns：
        quantized array
    """
    quantized_array= np.round(array/scale)
    quantized_array = np.maximum(quantized_array, -2**(data_width-1))
    quantized_array = np.minimum(quantized_array, 2**(data_width-1)-1)
    return quantized_array

def get_x_y_cuts(data, n_lines=1):
    """Find and put the similar pixels in a array.
    Args:
        data: image pixels data, 2-D array.
        n_lines: line of number. we use 1 line data ONLY.
    Returns:
        a list which single element is the axis range.
   """
    w, h = data.shape
    visited = set()
    q = queue.Queue()
    offset = [(-1, -1), (0, -1), (1, -1), (-1, 0),
              (1, 0), (-1, 1), (0, 1), (1, 1)]
    cuts = []
    for y in range(h):
        for x in range(w):
            x_axis = []
            y_axis = []
            if data[x][y] < 200 and (x, y) not in visited:
                q.put((x, y))
                visited.add((x, y))
            while not q.empty():
                x_p, y_p = q.get()
                for x_offset, y_offset in offset:
                    x_c, y_c = x_p + x_offset, y_p + y_offset
                    if (x_c, y_c) in visited:
                        continue
                    visited.add((x_c, y_c))
                    try:
                        if data[x_c][y_c] < 200:
                            q.put((x_c, y_c))
                            x_axis.append(x_c)
                            y_axis.append(y_c)
                    except:
                        pass
            if x_axis:
                min_x, max_x = min(x_axis), max(x_axis)
                min_y, max_y = min(y_axis), max(y_axis)
                if max_x - min_x > 3 and max_y - min_y > 3:
                    cuts.append([min_x, max_x + 1, min_y, max_y + 1])
    if n_lines == 1:
        cuts = sorted(cuts, key=lambda x: x[2])
        pr_item = cuts[0]
        count = 1
        len_cuts = len(cuts)
        new_cuts = [cuts[0]]
        pr_k = 0
        for i in range(1, len_cuts):
            pr_item = new_cuts[pr_k]
            now_item = cuts[i]
            if not (now_item[2] > pr_item[3]):
                new_cuts[pr_k][0] = min(pr_item[0], now_item[0])
                new_cuts[pr_k][1] = max(pr_item[1], now_item[1])
                new_cuts[pr_k][2] = min(pr_item[2], now_item[2])
                new_cuts[pr_k][3] = max(pr_item[3], now_item[3])
            else:
                new_cuts.append(now_item)
                pr_k += 1
        cuts = new_cuts
    return cuts

def get_image_cuts(image, dir=None, is_data=False, n_lines=1, data_needed=False, count=0,QUAN = False):
    """Cut the image, find relevant small pieces.
    Args:
        image: can input image or pixel array.
        dir: the small pieces image save dir.
        is_data: decide the type of input image.
        data_needed: decide whether to return pixel array.
        count: the number of small pieces 
        QUAN: decide whether to quantify
    Returns:
        the number of small pieces image(for Debug) or
        small pieces image pixel array
    """ 
    
    if is_data:
        data = image
    else:
        data = cv2.imread(image, 2)
    cuts = get_x_y_cuts(data, n_lines=n_lines)
    image_cuts = None
    for i, item in enumerate(cuts):
        count += 1
        max_dim = max(item[1] - item[0], item[3] - item[2])
        new_data = np.ones((int(1.4 * max_dim), int(1.4 * max_dim))) * 255
        x_min, x_max = (
            max_dim - item[1] + item[0]) // 2, (max_dim - item[1] + item[0]) // 2 + item[1] - item[0]
        y_min, y_max = (
            max_dim - item[3] + item[2]) // 2, (max_dim - item[3] + item[2]) // 2 + item[3] - item[2]
        new_data[int(0.2 * max_dim) + x_min:int(0.2 * max_dim) + x_max, int(0.2 * max_dim) +
                 y_min:int(0.2 * max_dim) + y_max] = data[item[0]:item[1], item[2]:item[3]]

        standard_data = cv2.resize(new_data, (SIZE, SIZE))
        cv2.imwrite(dir + str(count) + ".jpg", standard_data)
        if not data_needed:
            cv2.imwrite(dir + str(count) + ".jpg", standard_data)
        if data_needed:
            data_flat = np.reshape(standard_data, (1, SIZE*SIZE))
            data_flat = (255 - data_flat) / 255

            if QUAN == True:
                data_flat = quantized_np(data_flat,SCALE,data_width=8)
            else:
                pass
            
            if image_cuts is None:
                image_cuts = data_flat
            else:
                image_cuts = np.r_[image_cuts, data_flat]
    if data_needed:
        return image_cuts
    return count

def image_cut(quan):
    """Get a image and use get_image_cuts function pre-process it
    Args:
        quan: decide whether to quantify
    Returns:
        small image pixel array
    """
    # print("Please enter image source:")
    # mode = input("C(camera) or F(file)") or "F"
    mode = "F" # the defualt input image source
    if mode == "C":
        cap = cv2.VideoCapture(0) 
        _ , img = cap.read()
    elif mode == "F":
        img = cv2.imread(img_path, 2)
    else:
        print("Illegal source! Please run program again")
        sys.exit()
    image_cuts = get_image_cuts(
        img, dir="./img_cut_results/cut", is_data=True, count=0, data_needed=True, QUAN=quan)
    return image_cuts

    
#===============================================
#                 hw init
#==============================================
#IP_base_address
IP_BASE_ADDRESS    =  0x43C00000
ADDRESS_RANGE      = 0x180

XDETECTION_ACC_CRTL_BUS_ADDR_AP_CTRL            =0x00
XDETECTION_ACC_CRTL_BUS_ADDR_GIE                =0x04
XDETECTION_ACC_CRTL_BUS_ADDR_IER                =0x08
XDETECTION_ACC_CRTL_BUS_ADDR_ISR                =0x0c
XDETECTION_ACC_CRTL_BUS_ADDR_INPUT_OFFSET_DATA  =0x10
XDETECTION_ACC_CRTL_BUS_ADDR_OUTPUT_OFFSET_DATA =0x18
XDETECTION_ACC_CRTL_BUS_ADDR_WEIGHT_OFFSET_DATA =0x20
XDETECTION_ACC_CRTL_BUS_ADDR_BETA_OFFSET_DATA   =0x28
XDETECTION_ACC_CRTL_BUS_ADDR_INFM_NUM_DATA      =0x30
XDETECTION_ACC_CRTL_BUS_ADDR_OUTFM_NUM_DATA     =0x38
XDETECTION_ACC_CRTL_BUS_ADDR_KERNEL_SIZE_DATA   =0x40
XDETECTION_ACC_CRTL_BUS_ADDR_KERNEL_STRIDE_DATA =0x48
XDETECTION_ACC_CRTL_BUS_ADDR_TM_DATA            =0x50
XDETECTION_ACC_CRTL_BUS_ADDR_TN_DATA            =0x58
XDETECTION_ACC_CRTL_BUS_ADDR_TR_DATA            =0x60
XDETECTION_ACC_CRTL_BUS_ADDR_TC_DATA            =0x68
XDETECTION_ACC_CRTL_BUS_ADDR_MLOOPS_DATA        =0x70
XDETECTION_ACC_CRTL_BUS_ADDR_NLOOPS_DATA        =0x78
XDETECTION_ACC_CRTL_BUS_ADDR_LAYERTYPE_DATA     =0x80
XDETECTION_ACC_CRTL_BUS_ADDR_M_DATA             =0x88

def HMC_Init_EX(In_Address,Out_Address,Weight_offset,Beta_offset,InFM_num,OutFM_num,
                 Kernel_size,Kernel_stride,
                 TM,TN,TR,TC,
                 mLoops,nLoops,LayerType,
                 WEIGHT_BASE,BETA_BASE,M):
   
    # mapping memory
    mmio = MMIO(IP_BASE_ADDRESS,ADDRESS_RANGE)
    
    while True:
        ap_idle =  (mmio.read(XDETECTION_ACC_CRTL_BUS_ADDR_AP_CTRL)>>2)&0x01
        if(ap_idle):
            break
    
    mmio.write(XDETECTION_ACC_CRTL_BUS_ADDR_INPUT_OFFSET_DATA,  In_Address)
    mmio.write(XDETECTION_ACC_CRTL_BUS_ADDR_OUTPUT_OFFSET_DATA, Out_Address)
    mmio.write(XDETECTION_ACC_CRTL_BUS_ADDR_WEIGHT_OFFSET_DATA, WEIGHT_BASE+Weight_offset*4)
    mmio.write(XDETECTION_ACC_CRTL_BUS_ADDR_BETA_OFFSET_DATA,   BETA_BASE+Beta_offset*4)

    mmio.write(XDETECTION_ACC_CRTL_BUS_ADDR_INFM_NUM_DATA,      InFM_num)
    mmio.write(XDETECTION_ACC_CRTL_BUS_ADDR_OUTFM_NUM_DATA,     OutFM_num)
    mmio.write(XDETECTION_ACC_CRTL_BUS_ADDR_KERNEL_SIZE_DATA,   Kernel_size)
    mmio.write(XDETECTION_ACC_CRTL_BUS_ADDR_KERNEL_STRIDE_DATA, Kernel_stride)

    mmio.write(XDETECTION_ACC_CRTL_BUS_ADDR_TM_DATA,        TM)
    mmio.write(XDETECTION_ACC_CRTL_BUS_ADDR_TN_DATA,        TN)
    mmio.write(XDETECTION_ACC_CRTL_BUS_ADDR_TR_DATA,        TR)
    mmio.write(XDETECTION_ACC_CRTL_BUS_ADDR_TC_DATA,        TC)
    mmio.write(XDETECTION_ACC_CRTL_BUS_ADDR_MLOOPS_DATA,    mLoops)
    mmio.write(XDETECTION_ACC_CRTL_BUS_ADDR_NLOOPS_DATA,    nLoops)
    mmio.write(XDETECTION_ACC_CRTL_BUS_ADDR_LAYERTYPE_DATA, LayerType)
    mmio.write(XDETECTION_ACC_CRTL_BUS_ADDR_M_DATA,         M)

    mmio.write(XDETECTION_ACC_CRTL_BUS_ADDR_GIE,    0)
    mmio.write(XDETECTION_ACC_CRTL_BUS_ADDR_AP_CTRL,1)
    while True:
        ap_done =  (mmio.read(XDETECTION_ACC_CRTL_BUS_ADDR_AP_CTRL)>>1)&0x01
        if(ap_done):
            break


#==============================================
#                  hw  control
#==============================================
def calculator_ps(img_base_buffer):
         
    TR=0
    TC=0
    TM=0
    TN=0
    offset_index=0
    woffset = 0
    boffset = 0

    img_reorg_buffer = np.zeros((64, 8, 8), dtype=np.int32)

    for i in range(0,4):
       
        if i == 0:

            TR = 32
            TC = 32
            TM = 32
            TN = 1

            mLoops = 1
            nLoops = 1
    
            start_time = time.time()
            HMC_Init_EX(int(in_ptr[i]),int(out_ptr[i]),woffset,boffset,1,32,
                5,1,TM,TN,TR,TC,mLoops,nLoops,0,
                WEIGHT_BASE,BETA_BASE,M_value[0])
            end_time = time.time()
            # print("Conv0 time:", end_time - start_time)

            woffset += weight_offset[offset_index]
            boffset += beta_offset[offset_index]  
            offset_index = offset_index+1

        elif i == 1:

            TR = 16
            TC = 16
            TM = 32
            TN = 4

            mLoops = 2
            nLoops = 8

            start_time = time.time()
            HMC_Init_EX(int(in_ptr[i]),int(out_ptr[i]),woffset,boffset,32,64,
                        5,1,TM,TN,TR,TC,mLoops,nLoops,0,
                        WEIGHT_BASE,BETA_BASE,M_value[1])
            end_time = time.time()
            # print("Conv2 time:", end_time - start_time)
            
            woffset += weight_offset[offset_index]
            boffset += beta_offset[offset_index]  
            offset_index = offset_index+1

        elif i == 2:

            for m in range(0, 64):
                for r in range(0, 8):
                    for c in range(0, 8):
                        img_reorg_buffer[m][r][c] = img_base_buffer[m * 64 + r * 8 + c]

            for r in range(0, 8):
                for c in range(0, 8):
                    for m in range(0, 64):
                        img_base_buffer[r * 512 + c * 64 + m] = img_reorg_buffer[m][r][c]

            TR = 1
            TC = 1
            TM = 32
            TN = 4
            mLoops = 1024
            nLoops = 1

            start_time = time.time()
            HMC_Init_EX(int(in_ptr[i]),int(out_ptr[i]),woffset,boffset,8*8*64,1024,
                        1,1,TM,TN,TR,TC,mLoops,nLoops,1,
                        WEIGHT_BASE,BETA_BASE,M_value[2])
            end_time = time.time()
            # print("FC4 time:", end_time - start_time)

        elif i == 3:

            start_time = time.time()
            for m in range(0,14):
                for n in range(0,1024):
                    if n==0:
                        tmp_add_result = beta_base_buffer[1120 + m]
                    else:
                        tmp_add_result = img_base_buffer[m]
                    
                    partial_mul = img_base_buffer[11264+n]*weight_base_buffer[4246304 + m * 1024 + n]

                    img_base_buffer[m] = partial_mul + tmp_add_result

                # print(m, img_base_buffer[m])
            
            end_time = time.time()
            # print("FC5 time:", end_time - start_time)
    print("FPGA_Accelerate_Completed!!")

    #===============================================
#                calculator
#==============================================
def md(l, x):
    """Multiplication and division.
    Args:
        l: String
        x: Operator
    Returns:
        a string which insert calculator result
    """
    a = l.index(x)
    if x == '*' and l[a + 1] != '-':
        k = float(l[a - 1]) * float(l[a + 1])
    elif x == '/' and l[a + 1] != '-':
        k = float(l[a - 1]) / float(l[a + 1])
    elif x == '*' and l[a + 1] == '-':
        k = -(float(l[a - 1]) * float(l[a + 2]))
    elif x == '/' and l[a + 1] == '-':
        k = -(float(l[a - 1]) / float(l[a + 2]))
    del l[a - 1], l[a - 1], l[a - 1]
    l.insert(a - 1, str(k))
    return l

def calculator(formula):
    """Calculator main function
    Args:
        formula: a string.
    Returns:
        Calculator result.
    
    """
    l = re.findall('([\d\.]+|/|-|\+|\*)', formula)
    sum = 0
    while 1:
        if '*' in l and '/' not in l:
            md(l, '*')
        elif '*' not in l and '/' in l:
            md(l, '/')
        elif '*' in l and '/' in l:
            a = l.index('*')
            b = l.index('/')
            if a < b:
                md(l, '*')
            else:
                md(l, '/')
        else:
            if l[0] == '-':
                l[0] = l[0] + l[1]
                del l[1]
            try:
                sum += float(l[0])
            except Exception as e:
                pass
            for i in range(1, len(l), 2):
                if l[i] == '+' and l[i + 1] != '-':
                    sum += float(l[i + 1])
                elif l[i] == '+' and l[i + 1] == '-':
                    sum -= float(l[i + 2])
                elif l[i] == '-' and l[i + 1] == '-':
                    sum += float(l[i + 2])
                elif l[i] == '-' and l[i + 1] != '-':
                    sum -= float(l[i + 1])
            break
    return sum

#==============================================
#                 main function
#==============================================
def inference():
    img = image_cut(quan=True)
    formula = ''
    for i in range(np.size(img, 0)):
        np.copyto(img_base_buffer[0:1024], img[i].astype(np.int32))
        calculator_ps(img_base_buffer)
        index = np.argmax(img_base_buffer[0:14], 0)
        formula += SYMBOL[index]
    print("The Handwritten Mathematical Expression is:", formula)
    result = calculator(formula)
    print("The Result is:", result)
    return formula, result

if __name__ == '__main__':
    main()
