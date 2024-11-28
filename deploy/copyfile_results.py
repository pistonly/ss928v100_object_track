import telnetlib
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime


def do_telnet(tn, finish, commands):

    for command in commands:
        tn.write(command.encode('ascii') + b'\n')
        time.sleep(2)
        # print(tn.read_very_eager())

    tn.write(b'\n')
    tn.read_until(finish.encode('ascii'))


def link_ss928(Host, username, password, finish):
    try:
        tn = telnetlib.Telnet(Host, port=23, timeout=10)
        tn.set_debuglevel(2)

        tn.read_until(b'localhost login: ')
        tn.write(username.encode('ascii') + b'\n')

        tn.read_until(b'Password: ')
        tn.write(password.encode('ascii') + b'\n')

        tn.read_until(finish.encode('ascii'))

        return tn
    except Exception as e:
        print(f"failed to connect {Host}: {e}")
        return None


if __name__ == '__main__':

    ss928_ip = ['11', '12', '13', '21', '22', '23', '31', '32', '33', '100']

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, nargs="+")
    args = parser.parse_args()
    if args.ip:
        ss928_ip = args.ip


    Host = '192.168.0.'
    username = 'root'
    password = ''
    finish = '~ #'

    # 获取当前时间
    now = datetime.datetime.now()
    # 格式化时间为可读形式
    now_str = now.strftime('%Y%m%d_%H%M%S')

    def execute_on_device(ip):

        target_dir_parent = f"/mnt/nfs/sot_results/{now_str}"
        target_dir = f"{target_dir_parent}/{ip}"

        commands = [
            # kill && mount nfs
            'pkill one_camera_yolo_track_2chns_1080p; mount -t nfs -o nolock 192.168.0.77:/e/nfs_share /mnt/nfs',
            f'''mkdir -p {target_dir_parent} && nohup sh -c "cp -r /mnt/data/sot {target_dir} &&rm -r /mnt/data/sot && mkdir -p /mnt/data/sot && cp /mnt/data/track_log.log {target_dir}/ && rm /mnt/data/track_log.log" > /root/copy_result.log 2>&1 &'''
        ]
        tn = link_ss928(Host + ip, username, password, finish)
        if tn:
            do_telnet(tn, finish, commands)
            tn.close()

    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(execute_on_device, ss928_ip)


