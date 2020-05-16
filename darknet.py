from common import convolutional, residual_block


def darknet53(inputs):
    inputs = convolutional(inputs, 32, (3, 3))
    inputs = convolutional(inputs, 64, (3, 3), downsample=True)

    for i in range(1):
        inputs = residual_block(inputs, 32, 64)

    inputs = convolutional(inputs, 128, (3, 3), downsample=True)

    for i in range(2):
        inputs = residual_block(inputs, 64, 128)

    inputs = convolutional(inputs, 256, (3, 3), downsample=True)

    for i in range(8):
        inputs = residual_block(inputs, 128, 256)

    route1 = inputs
    inputs = convolutional(inputs, 512, (3, 3), downsample=True)

    for i in range(8):
        inputs = residual_block(inputs, 256, 512)

    route2 = inputs
    inputs = convolutional(inputs, 1024, (3, 3), downsample=True)

    for i in range(4):
        inputs = residual_block(inputs, 512, 1024)

    route3 = inputs
    return route1, route2, route3