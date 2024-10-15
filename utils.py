def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    """
    加载预训练模型的参数到指定模型中。

    该函数会将提供的预训练模型参数（state_dict）加载到指定的模型中。
    可以通过前缀（prefix）来筛选出需要加载的参数。同时，该函数还支持忽略某些缺失的键而不报错，
    这是通过设置ignore_missing参数来实现的。

    参数:
        model: 要加载参数的模型。
        state_dict: 预训练模型的参数字典。
        prefix: 用于筛选state_dict中键的前缀。
        ignore_missing: 忽略报告的缺失键，支持用'|'分隔的多个键。

    返回:
        无返回值，但会打印出缺失的、多余的、以及被忽略的键的信息。
    """
    # 初始化记录缺失键、多余键和错误信息的列表
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # 复制state_dict以便内部函数_load_from_state_dict可以修改它
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # 定义一个加载函数，用于递归加载模型参数
    def load(module, prefix=''):
        # 初始化本地元数据，如果metadata为None，则初始化为空字典，否则获取与前缀对应的元数据
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        # 从状态字典中加载模块参数，遇到缺失或意外的键时进行记录
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        # 遍历模块的所有子模块，递归加载子模块的参数
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')    

    # 调用加载函数开始加载模型参数
    load(model, prefix=prefix)

    # 分类处理缺失的键，决定哪些键需要警告，哪些键被忽略
    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    # 更新缺失的键列表，只保留需要警告的键
    missing_keys = warn_missing_keys

    # 打印总结信息
    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))