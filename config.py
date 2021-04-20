class Config:
    def __init__(self):
        super(Config, self).__init__()
        self.ip = '0.0.0.0'
        self.public_ip = 'http://10.174.73.103'
        self.port = '5000'

        self.index_path = self.public_ip + ':' + self.port + '/'
        self.predict_path = self.index_path + 'predict/'
        self.result_path = self.predict_path + 'result'