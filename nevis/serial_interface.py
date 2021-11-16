from tkinter.constants import COMMAND
import serial
import math
from math import ceil
import tkinter as tk
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from timeit import default_timer as timer

# Parameters for the setup of the GUI
x_range = 1000
run_test_function = False

w = 0
scale_slider = 0
update_slider = 0
mse_text = 0

timestep = 0
hardware_timestep = 0

def run_serial_interface(
        model, 
        output_scale,
        out_depth, 
        run_model=True,
        sim_test_function = lambda x, t: math.sin(x * t),
        baud =2000000, 
    ):

    global input_data, ideal_data, hardware_data, sim_data, sim_test_data, hardware_test_data
    global w, update_slider, scale_slider, mse_text
    global run_test_function, sine_timestep, timestep

    sine_timestep = 0
    timestep = 0

    input_data = []
    ideal_data = []
    hardware_data = [0]
    sim_data = []
    sim_test_data = []
    hardware_test_data = []

    bytes_to_read = ceil(out_depth / 8.0)
    depth = out_depth

    serial_port = '/dev/tty.usbserial-FT4ZS6I31'
    ser = 0
    print("\nAttempting to open serial port...")
    while type(ser) == int: 
        try:
            ser = serial.Serial(serial_port, baudrate=baud, timeout=0.0005) # '''timeout=0.0005'''
            print('Opened serial port to device at ', ser.name)
        except:
            time.sleep(0.1)
            pass

    time.sleep(0.6)
    # Define the decoder function TODO - Pass this from the compiler.
    def decoder(input):
        return int(input*127)/127.0

    def twos_complementer(value):
        
        sign = (value < 0)
        value = abs(value)

        if sign:
            value = bin(value)
            value = value[2:]

            value = value.replace('1', '2')
            value = value.replace('0', '1')
            value = value.replace('2', '0')

            value = ('1')*(8 - len(value)) + value
            
            value = int(value, 2)
        
        return bytes([value])

    # Plot the input value
    def plot_input():
        global input_data, ideal_data, hardware_data, sim_data, sim_test_data, hardware_test_data
        global w, update_slider, scale_slider, mse_text
        global run_test_function, sine_timestep, timestep, hardware_timestep
        
        def shift_data(data, value):
            data[0:x_range-1] = data[1:x_range]
            data[x_range-1] = value
            return data

        def set_lines(lines, data):
            lines.set_xdata(range(0, len(data)))
            lines.set_ydata(data)

        input_val = w.get()
        ideal_val = decoder(input_val)

        x_range = scale_slider.get()

        for i in range(update_slider.get()):

            if run_test_function:
                input_val = sim_test_function(sine_timestep, 0.001)
                ideal_val = decoder(input_val)
                w.set(input_val)
                sine_timestep += 1
                if sine_timestep == 6283:

                    # Compute the RMSE for the sine wave period
                    errors = [(y-x)**2 for x,y in zip(sim_test_data[-6283:], hardware_test_data[-6283:])]
                    test_mse_val = sum(errors) / len(errors)
                    test_mse_val = math.sqrt(test_mse_val)
                    print("RMSE for the last period of the test function:", str("%.5f" % round(test_mse_val, 5)))
                    
                    # Compute a filtered RMSE to test overall behaviour
                    def chunker(seq, size):
                        # From https://stackoverflow.com/questions/434287/
                        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
                    
                    kernel_size = 100
                    errors = []
                    for groups in zip(chunker(sim_test_data, kernel_size), chunker(hardware_test_data, kernel_size)):
                        error_val = (sum(groups[1]) / len(groups[1]) - sum(groups[0]) / len(groups[0]))**2
                        errors.append(error_val)
                    test_mse_val = math.sqrt(sum(errors) / len(errors))
                    print("Filtered RMSE for the last period of the test function:", str("%.5f" % round(test_mse_val, 5)))
                    sim_test_data = []
                    hardware_test_data = []
                    sine_timestep = 0
                         
            output_num = int(input_val * 127)
            i_x = twos_complementer(output_num)
            
            try:
                ser.write(i_x)
                hardware_timestep += 1
            except:
                return
            
            rx_data = ser.read(size=bytes_to_read)
            
            #ser.flush()

            if len(rx_data) == bytes_to_read:
                hardware_val = int.from_bytes(rx_data, byteorder="big", signed=True)
                hardware_val = hardware_val / (2**output_scale)

                temp_vals = []
                if run_model:
                    while timestep != hardware_timestep:
                        # Calculate the value from the software simulation
                        model_input_val = output_num / 128
                        model.run_timestep(model_input_val, timestep)
                        timestep += 1
                        if run_test_function:
                            sim_test_data.append(model.output)
                        temp_vals.append(model.output)

                if (len(input_data) < x_range):
                    hardware_data.append(hardware_val)
                    input_data.append(input_val)
                    ideal_data.append(ideal_val)
                    if run_model:
                        sim_data += temp_vals
                else:
                    input_data = shift_data(input_data, input_val)
                    ideal_data = shift_data(ideal_data, ideal_val)
                    hardware_data = shift_data(hardware_data, hardware_val)
                    if run_model:
                        for val in temp_vals:
                            sim_data = shift_data(sim_data, val)

                if run_test_function:
                    hardware_test_data.append(hardware_val)
            else:
                print("WARNING: Rx data dropped")

            time.sleep(0.001)
        
        set_lines(lines_input, input_data)
        set_lines(lines_ideal, ideal_data)
        set_lines(lines_hardware, hardware_data)
        set_lines(lines_simulation, sim_data)
        
        ax.set_xlim(0, x_range)

        canvas.draw()
        root.after(1, plot_input)

    def run_test():
        global run_test_function
        run_test_function = True

    def stop_test():
        global run_test_function, sine_timestep
        run_test_function = False
        sine_timestep = 0

    # Setup GUI
    root = tk.Tk()
    root.title('NeVIS Prototype - Neural Data')
    root.configure(background='white')
    resolution = [900, 600]

    root.geometry(str(resolution[0]) + 'x' + str(resolution[1]))

    # Setup plt
    fig = plt.Figure(dpi=100)
    ax = fig.add_subplot(111)

    ax.set_title('NeVIS Prototype - Output Data Plot')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_xlim(0, x_range)
    ax.set_ylim(-1.2, 1.2)

    lines_input = ax.plot([], [], label='Input')[0]
    lines_ideal = ax.plot([], [], label='Ideal')[0]
    lines_hardware = ax.plot([], [], label='FPGA Output')[0]
    lines_simulation = ax.plot([], [], label='Nengo Output')[0]
    leg = ax.legend()

    # Setup Tk drawing area
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().place(x=10, y=10, width=750, height=550)
    canvas.draw()

    w = tk.Scale(root, from_=1.0, to=-1.0, resolution=0.001, length=500)
    w.place(x=resolution[0]-70, y=50)
    w.set(1.0)

    scale_slider = tk.Scale(root, from_=100, to=10000, resolution=1, length=500)
    scale_slider.place(x=resolution[0]-150, y=50)
    scale_slider.set(x_range)

    update_slider = tk.Scale(root, from_=1, to=64, resolution=1, length=500)
    update_slider.place(x=resolution[0]-200, y=50)
    update_slider.set(4)

    test_fx_button = tk.Button(root, text="Run Test f(x)", command=lambda: run_test())
    test_fx_button.pack(side='bottom')

    test_fx_button = tk.Button(root, text="Stop Test f(x)", command=lambda: stop_test())
    test_fx_button.pack(side='bottom')

    exit_button = tk.Button(root, text="Exit", command=root.destroy)
    exit_button.place(x=15, y=15)

    mse_text = tk.StringVar()
    mse_text.set("MSE for last 100 timesteps, x1000: ")
    mse_label = tk.Label(root, textvariable = mse_text).place(x=50, y=resolution[1]-50)

    root.update()
    root.after(1, plot_input)
    root.tk.call('tk', 'scaling', 2.0)
    root.mainloop()