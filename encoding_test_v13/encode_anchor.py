import subprocess
import os
import argparse
import re


def get_class_group(yuv_filename: str):
    if "3840x2160" in yuv_filename:
        return "ClassA"
    elif "1920x1080" in yuv_filename:
        return "ClassB"
    elif "832x480" in yuv_filename:
        return "ClassC"
    elif "416x240" in yuv_filename:
        return "ClassD"
    elif "1280x720" in yuv_filename:
        return "ClassE"
    return "None"


def get_level(yuv_filename: str):
    if "RaceHorses_832x480" in yuv_filename:
        return "3"
    if "RaceHorses_416x240" in yuv_filename:
        return "2"

    if "3840x2160" in yuv_filename:
        return "5.1"
    elif "1920x1080" in yuv_filename:
        return "4.1"
    elif "832x480" in yuv_filename:
        return "3.1"
    elif "416x240" in yuv_filename:
        return "2.1"
    elif "1280x720" in yuv_filename:
        return "4"
    return "3.1"


def parse_file_name(yuv_file: str):  # 输入形式：BQSquare_416x240_60(fps)_8(bit)_420.yuv
    # seq_name = yuv_file.split('.')[0]
    # pattern = r'\d+x\d+'
    # shape = re.findall(pattern, yuv_file)[0]
    # w, h = shape.split('x')
    seq_name, shape, fps, bit_depth, yuv_format, frame_count = yuv_file.split('_')
    w, h = shape.split('x')

    return yuv_file, w, h, fps, bit_depth, yuv_format, frame_count


def make_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', type=str, action='append')
    parser.add_argument('--frame', type=str)
    args = parser.parse_args()

    # 基础参数设置
    yuv_list = args.list  #
    TemporalSubsampleRatio = 8
    frame = 10
    if args.frame != "all":
        frame = int(args.frame) * TemporalSubsampleRatio

    encoder = "EncoderApp_Anchor_v13.exe"  #
    cfg_file = "encoder_intra_vtm.cfg"
    encoding_folder = r'encoding_files/'
    summary_folder = 'summary\\'  #
    output_folder = r'enc_output/'

    QPs = ['37', '32', '27', '22']
    # 对指定的序列进行 4 个QP值编码，此外需要针对是否进行基准测试进行判断
    file_dir = f'anchor'
    make_dir(file_dir)
    make_dir(os.path.join(file_dir, encoding_folder))
    make_dir(os.path.join(file_dir, output_folder))
    make_dir(os.path.join(file_dir, summary_folder))

    for yuv_file in yuv_list:
        class_group = get_class_group(yuv_file)
        if class_group == "None":
            continue
        # 获取 level
        level = get_level(yuv_file)

        file_name_without_postfix, width, height, fps, bit_depth, yuv_format, frame_count = parse_file_name(yuv_file)  # 获取无后缀的序列名，宽度，高度
        yuv_file_path = f"../test_sequences/{class_group}/{file_name_without_postfix}.yuv"
        if args.frame == "all":
            frame = frame_count

        # 对 4 个QP值进行编码
        for QP in QPs:
            bitstream_file = os.path.join(file_dir, encoding_folder, f'{file_name_without_postfix}_QP{QP}.bin')
            reconfile_file = os.path.join(file_dir, encoding_folder, f'{file_name_without_postfix}_QP{QP}.yuv')
            summary_file = os.path.join(file_dir, summary_folder, f'{file_name_without_postfix}.txt')
            output_file = os.path.join(file_dir, output_folder, f'{file_name_without_postfix}_QP{QP}.txt')

            # 进行编码
            with open(output_file, mode='wb') as fb:
                command = [
                    f'{encoder}',
                    '-c', 'encoder_intra_vtm.cfg',
                    f'--InputFile={yuv_file_path}',
                    f'--BitstreamFile={bitstream_file}',
                    f'--ReconFile={reconfile_file}',
                    f'--SummaryOutFilename={summary_file}',
                    f'--InputBitDepth={bit_depth}',
                    f'--InputChromaFormat={yuv_format}',
                    f'--FrameRate={fps}',
                    f'--SourceWidth={width}',
                    f'--SourceHeight={height}',
                    f'--FramesToBeEncoded={frame}',
                    f'--Level={level}',
                    f'--QP={QP}',
                    f'--TemporalSubsampleRatio={TemporalSubsampleRatio}',
                ]

                print(command)
                subprocess.run(command, stdout=fb)




