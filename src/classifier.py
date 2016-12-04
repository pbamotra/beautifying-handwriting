import vgg
import scipy
from utils import *
from pandas import *
import tensorflow as tf
from collections import Counter
from sklearn.utils import shuffle
from collections import defaultdict
from argparse import ArgumentParser
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegressionCV

STYLE_LAYERS = ('relu1_1','relu2_1', 'relu3_1')
VGG_PATH = '../models/imagenet-vgg-verydeep-19.mat'
MODEL_FILE = '../models/font-classifier-f3-100-{}.pkl'
TRAINED_PCA = '../models/pca-f3-100-{}.pkl'


def main(network, chars=None, load_model=False, n_fonts=None, target_font=None, test_image=None, test_label=None):

    print 'Choosing', n_fonts, 'fonts'

    fonts_data = read_data(verbose=False)
    data = get_font_image_by_char(font_data=fonts_data, n_subset=n_fonts, char=chars)

    style_features = defaultdict(dict)

    print "Load model is", 'enabled' if load_model else 'disabled'

    if not load_model:
        for font_id, font in enumerate(data):
            print 'Processing font', font_id + 1

            for ch_id, font_ch in enumerate(font):
                print '\t\tProcessing character', font_id + 1, '/', ch_id + 1
                image = np.array(list(font_ch.getdata())).reshape(64, 64, 3)
                vgg_features = get_style_features(image, network)
                style_features[font_id][ch_id] = np.concatenate(vgg_features).ravel().tolist()

        x = []
        y = []

        for font_id, ch_id_features in style_features.iteritems():
            for ch_id, features in ch_id_features.iteritems():
                x.append(features)
                y.append(1 if font_id == target_font else 0)

        x = np.array(x)
        y = np.array(y)

        print 'Input shape: ', x.shape, 'Label shape: ', y.shape
        print 'y:', Counter(y)

        n_components = n_fonts * len(chars) - 1
        pca = PCA(n_components=n_components, whiten=True)
        pca_x = pca.fit_transform(x, y)

        print 'PCA n={} cover {} variance'.format(n_components, sum(pca.explained_variance_ratio_))

        x, y = shuffle(pca_x, y, random_state=42)
        clf = LogisticRegressionCV(cv=len(chars), class_weight='balanced')
        clf.fit(x, y)
        joblib.dump(clf, MODEL_FILE.format(target_font))
        joblib.dump(pca, TRAINED_PCA.format(target_font))

        print 'Confusion matrix'
        print confusion_matrix(y, clf.predict(x))
    else:
        if test_image is None or test_label is None:
            print 'Please provide test image and test label'
            return

        clf = joblib.load(MODEL_FILE.format(test_label))
        pca = joblib.load(TRAINED_PCA.format(test_label))

        # print 'Test input image shape', test_image.shape
        test_image = np.concatenate(get_style_features(test_image, VGG_PATH)).ravel().reshape(-1, 1).T.tolist()
        #print 'Test input image after VGG processing looks like', len(test_image), 'x 1'

        x = pca.transform(test_image)

        print 'Classifier prediction is (0=incorrect_pred, 1=correct_pred)', clf.predict(x)
        print 'Prediction probability is', clf.predict_proba(x)
        #print 'Actual label was', test_label


def get_style_features(input_data, network):
    style_shape = (1,) + input_data.shape
    g = tf.Graph()

    ch_all_styles = []
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=style_shape)
        net, mean_pixel = vgg.net(network, image)
        style_pre = np.array([vgg.preprocess(input_data, mean_pixel)])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={image: style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            ch_all_styles.append(gram.ravel())

    return ch_all_styles

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--network',
                        dest='network', help='path to network parameters (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--n_fonts', dest='n_fonts', help='number of fonts to use for training',
                        default=30, required=True)
    parser.add_argument('--chars', dest='chars', help='characters to use for training e.g. a,b,c,d',
                        default=4, required=True)
    parser.add_argument('--load', dest='load', help='Load a saved model', default=False, action='store_true')
    parser.add_argument('--target_id', dest='target_id', help='font number of the target font (0 index)', required=True)

    options = parser.parse_args()
    n_fonts, network = int(options.n_fonts), options.network
    load_model = options.load
    chars = list(options.chars.split(','))
    target_font = int(options.target_id)

    test_image = '../data/run/A/style/19_A.png'
    test_image = './final/19_A.pnghave_0_5__0_5_f3_normal.png'
    test_image = scipy.misc.imread(test_image).astype(np.float)
    test_label = 19

    # for img in os.listdir('../data/images/'):
    #     if 'a' in img:
    #         print 'Using image', img
    #         test_image = '../data/images/' + img
    #         test_image = scipy.misc.imread(test_image).astype(np.float)
    #         test_label = 19

    main(network, chars, load_model, n_fonts, target_font, test_image=test_image, test_label=test_label)
    print '-' * 150
