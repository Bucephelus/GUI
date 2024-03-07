import time
import serial

ser = serial.Serial('COM3', 9600)  
time.sleep(2)

dx = 0
# x1 = bin(x)
# dx = f"{x}\n".encode()
dy = 50

while True:
    ser.write(f"{dx},{dy}\n".encode())
    time.sleep(0.05)
    # ser.write(dx)
    print(dx,dy)
