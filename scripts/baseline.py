import os

INIT = '../'
DATA_PATH = INIT + 'data/run/'
STYLE_FOLDER = '/style'
IMAGE_FOLDER = '/image'
N_ALPHA = 4

if __name__ == '__main__':
    walks = os.walk(DATA_PATH)

    result = [[] for _ in xrange(2 * N_ALPHA)]
    i = -1

    for walk_id, walk in enumerate(walks, start=1):
        # print walk_id, walk
        if len(walk[1]) == 0:
            for image in filter(lambda filename: not filename.startswith('.'), walk[-1]):
                if 'image' in walk[0]:
                    i += 1
                # print i
                result[i].append(walk[0] + '/' + image)

    # print result
    command = 'python neural_style.py --content {} --styles {} --output final/{} ' \
              '--network ../models/imagenet-vgg-verydeep-19.mat --iterations 500 --content-weight 0.8 ' \
              '--style-weight 0.2 --checkpoint-output chkpoint_images/{}/{}%s.png --checkpoint-iterations 20'

    for i in xrange(len(result) - 1):
        combination = result[i]
        combination2 = result[i + 1]
        # print combination, combination2

        for content_file in [combination[0], combination2[0]]:
            # print content_file
            content_filename_split = content_file.split('_')
            font = content_filename_split[0].split('/')[3]

            for style_file in combination2[1:]:
                print command.format(content_file,
                                     style_file,
                                     style_file.split('/')[-1] + content_file.split('/')[-1],
                                     font,
                                     style_file.split('/')[-1] + content_file.split('/')[-1])