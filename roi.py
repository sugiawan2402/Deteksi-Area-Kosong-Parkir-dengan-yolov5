

import cv2

c=True

def show_roi_area(im,object1, object2,line_thickness=3):
    global c
    RoIs = object2
    B, A= 0, 0

    # tempat parkir   : [30, 265, 310, 510]
    # kendaraan       : [67, 324, 301, 508]
    #
    # tempat parkir   : [915, 265, 1250, 510]
    # kendaraan       : [922, 344, 1218, 503]
    # color = (0,0,255)
    spaceCounter = 0
    for a in RoIs: #tempat parkir
        for b in object1: #kendaraan
            # print('kendaraan     : ',b)
            # print('tempat parkir : ',a)
            if a[0] < b[0] and a[1] < b[1] and a[2] > b[2] and a[3] > b[3]:
                A=A+1
                color =(0,0,255)
                spaceCounter += 1
                print(spaceCounter)
            else:
                color = (0, 255, 0)
            cv2.rectangle(im, (a[0], a[1]), (a[2], a[3]), color, 3)
        if A==0:
            B=B+1
            # color = (0, 255, 0)
    
    #msg='jumlah tempat parkir : ' + str(len(object2)) + '  tempat parkir kosong : ' +str(B)
    msg = f'Free: {len(RoIs)}/{str(B)}'
    # print(msg)
    
    cv2.putText(im, msg, (0,50), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
    return  c







