import string
# import easyocr
# import csv
import re
import pandas as pd
from paddleocr import PaddleOCR

# Initialize the OCR reader
# reader = easyocr.Reader(['en'], gpu=False)
ocr = PaddleOCR(lang='en')

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    'B': '8',
                    'E': '8',
                    'Z': '2',
                    'T': '7',
                    'H': '8',
                    'D': '0',
                    'Q': '0',
                    }

 

dict_int_to_char = {'0': 'D',
                    '1': 'A',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '8': 'B',
                    '7': 'T',
                    }

# def write_csv(results, output_path):
#     with open(output_path, 'w') as f:
#         writer = csv.writer(f)

#         writer.writerow(['car_id','time','license_plate_score','license_plate_number'])

#         for row in results:
#             writer.writerow([row[0],row[1]['time'],row[1]['license_plate_score'],row[1]['license_plate_number']])
        
#     f.close()

# write into xlsx file
def write_xlsx(results, output_path):
    car_info_list = []
    for car_id, car_info in results.items():
        car_info_list.append({
            'license_plate_number': car_info['license_plate_number'],
            'timestamp': car_info['timestamp']
        })
    df = pd.DataFrame(car_info_list)
    df.to_excel(output_path, index=False)

def to_chars(text):
    str = ''
    for ch in text:
        if ch in dict_int_to_char:
            str+=dict_int_to_char[ch]
        else:
            str+= ch
    return str

def to_ints(text):
    str = ''
    for ch in text:
        if ch in dict_char_to_int:
            str+=dict_char_to_int[ch]
        else:
            str+= ch
    return str

def is_all_chars(text):
    for ch in text:
        if not (ch in string.ascii_uppercase or ch in dict_int_to_char.keys()):
            return False
    return True

def license_complies_format(text):
    if len(text) < 9 or len(text)>10:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       ((text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) or \
       text[0] == 'D' and text[1]=='L' and text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and \
       is_all_chars(text[4:-4]) and \
       (text[-1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[-1] in dict_char_to_int.keys()) and \
       (text[-2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[-2] in dict_char_to_int.keys()) and \
       (text[-3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[-3] in dict_char_to_int.keys()) and \
       (text[-4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[-4] in dict_char_to_int.keys()) :
        return True
    else:
        return False



def format_license(text):
    if len(text)<9 or len(text)>10:
        return None

    if text[:2] == 'OL' or  text[:2] == 'GL' or text[:2] == 'QL':
        text = 'DL' + text[2:]
    license_plate_ = ''
    license_plate_ += to_chars(text[0:2])
    if text[:2] == 'DL' :
        license_plate_ += to_ints(text[2])
        license_plate_ += to_chars(text[3])
    else :
        license_plate_ += to_ints(text[2:4])
    license_plate_ += to_chars(text[4:-4])
    license_plate_ += to_ints(text[-4:])

    return license_plate_


def read_license_plate(license_plate_crop):
    # detections = reader.readtext(license_plate_crop)

    # for detection in detections:
    #     bbox, text, score = detection

    #     text = text.upper()
    #     text = re.sub(r"[^A-Z0-9]", "", text)

    #     text = format_license(text)
    #     if text is not None and license_complies_format(text):
    #     # if text:
    #         return text, score
    #         # return text, score

    # using paddleocr
    result = ocr.ocr(license_plate_crop, cls = False)
    curr_result = result[0]
    if curr_result is not None:
        recognised_text = [line[1][0] for line in curr_result][0]
        score = [line[1][1] for line in curr_result][0]
        recognised_text = recognised_text.upper()
        recognised_text = re.sub(r"[^A-Z0-9]", "", recognised_text)
        recognised_text = format_license(recognised_text)
        if recognised_text is not None and license_complies_format(recognised_text):
            return recognised_text, score

    return None, None

def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1



