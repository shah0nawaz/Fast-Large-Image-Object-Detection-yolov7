import os


class PreProcess:
    def __init__(self, crops_path, crops_list_file):
        self.crops_path = crops_path
        self.crops_list_file = crops_list_file

    def generate_valid_list(self):
        img_files = os.listdir(self.crops_path)
        img_files = sorted(img_files)
        f = open(self.crops_list_file, 'a')
        for img_file in img_files:
            if not img_file.endswith('.jpg'):
                continue
            f.write(f'{self.crops_path}{img_file}\n')
        f.close()
