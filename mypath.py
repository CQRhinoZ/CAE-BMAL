class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'fundus':
            return '/ai/sjq/dataset/Fundus/Domain1/train/ROIs/image/'  # foler that contains leftImg8bit/
        elif database == 'wae':
            return '/ai/sjq/mylab/VAE/wae_img/'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
