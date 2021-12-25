import struct as st


def save_data(tensor, path, is_act=False, to_int=False, to_hex=False, output_dir=None, q=0.0):
    def identity(x):
        return x

    def convert_int(x):
        return int(x)

    def convert_hex(x):
        return '%X' % st.unpack('H', st.pack('e', x))

    def convert_act(x):
        return round((x * (2 ** q)).item())

    print(f'Saving {path}')
    dir_name = output_dir

    type_cast = identity
    if to_int:
        type_cast = convert_int
    elif to_hex:
        type_cast = convert_hex
    elif is_act:
        type_cast = convert_act

    path = f'{dir_name}/{path}'
    with open(f'{path}.txt', 'w') as f:
        print('\n'.join(
            f'{type_cast(num.item())}'
            for num in tensor.half().view(-1)
        ), file=f)
