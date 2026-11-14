def DeNormalizeData(data, data_min, data_max):
    return data * (data_max - data_min) + data_min
