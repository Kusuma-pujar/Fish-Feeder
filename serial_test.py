import serial
import time
data = serial.Serial(
                    'COM3',
                    baudrate = 9600,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS,
                    timeout=1
                    )

def Read():
  print("reading")
  d = []
  while True:
    Data = data.readline()
    Data = Data.decode('utf-8', 'ignore')
    Data = Data.split(',')
    print(Data)
    if len(Data)  == 4:
      d.append(float(Data[0]))
      d.append(float(Data[1]))
      d.append(float(Data[2]))
      d.append(float(Data[3].replace('\r\n', '')))
      break
  print(d)
  return d
