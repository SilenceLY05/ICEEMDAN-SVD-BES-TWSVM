import csv

def find_max_y_in_range(filename, x_min, x_max):
    max_y = None
    max_y_x = None
    
    # Open the file and read its contents
    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file)
        
        # Skip the header
        next(reader)
        
        # Loop through the remaining rows
        for row in reader:
            x, y = map(float, row)
            
            # Check if the current X value is within the specified range
            if x_min <= x <= x_max:
                if max_y is None or y > max_y:
                    max_y = y
                    max_y_x = x
    
    return max_y, max_y_x

# Example usage:
# Replace the file path and range with actual values
filename = r'G:\毕业论文文件\OTDR_Data\新建文件夹2\adjusted_data_new2.txt'  # Adjust with the actual path
x_min = 19.0  # Adjust with the actual minimum X value of the range
x_max = 20.0  # Adjust with the actual maximum X value of the range

max_y, max_y_x = find_max_y_in_range(filename, x_min, x_max)

print(f'Maximum Y value within the range ({x_min}, {x_max}): {max_y} at X value {max_y_x}')
