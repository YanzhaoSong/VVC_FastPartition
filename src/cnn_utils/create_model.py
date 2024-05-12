import timm

from cnn_models import MyNet


def create_model(
        args,
        from_timm: bool,
):
    """
    :param args:
    :param model_name: 模型名字
    :param in_channels: 输入的通道数
    :param num_classes: 分类数
    :param from_timm: 模型是否来自于 timm
    :param kwargs: 其他相关的参数
    :return:
    """

    if from_timm is True:
        model = timm.create_model(
            model_name=args.model_name,
            pretrained=False,
            in_chans=args.in_channels,
            num_classes=args.num_classes,
        )
    else:
        if args.model_name == 'MyNet':
            model = MyNet(args.in_channels, args.num_classes)
        else:
            raise RuntimeError('Unknown model (%s)' % args.model_name)
    model.model_name = args.model_name

    return model


if __name__ == "__main__":
    from types import SimpleNamespace
    from torchsummary import summary

    model_args = SimpleNamespace()
    model_args.model_name = 'MyNet'
    model_args.in_channels = 1
    model_args.num_classes = 6

    model = create_model(model_args, from_timm=False)

    print(model)
    summary(model, (1, 32, 32))