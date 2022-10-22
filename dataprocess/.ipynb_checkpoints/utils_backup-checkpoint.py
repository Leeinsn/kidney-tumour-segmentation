def trans_roi(roi, shape, origin_shape=(128, 248, 248)):
    # 三个维度的ratio
    ratio = [s0/s1 for s0, s1 in zip(shape, origin_shape)]
    new_roi = []
    for item in roi:
        ordi = [int(v) for v in item]
        trans_ordi = [ordi[0]*ratio[0], ordi[1]*ratio[0], 
                      ordi[2]*ratio[1], ordi[3]*ratio[1], 
                      ordi[4]*ratio[2], ordi[5]*ratio[2]]
        trans_ordi = np.round(np.array(trans_ordi))
        # 保留部分冗余
        # trans_ordi[0:6:2] -= 3
        # trans_ordi[1:6:2] += 3
        new_roi.append(trans_ordi.astype('int'))
    return new_roi

def get_roi_data(roi_ann_path):
    with open(roi_ann_path, 'r') as f:
        data = f.readlines()
    f.close()
    ordi = []
    # length is 6, type 'str'
    for item in data:
        item = item.strip().split(' ')
        ordi.append(item)
    return ordi