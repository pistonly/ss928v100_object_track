import telnetlib
import time
import json
import argparse


def do_telnet(tn, finish, commands):

    for command in commands:
        tn.write(command.encode('ascii') + b'\n')
        time.sleep(2)
        # print(tn.read_very_eager())

    tn.write(b'\n')
    tn.read_until(finish.encode('ascii'))


def link_ss928(Host, username, password, finish):
    tn = telnetlib.Telnet(Host, port=23, timeout=10)
    tn.set_debuglevel(2)

    tn.read_until(b'localhost login: ')
    tn.write(username.encode('ascii') + b'\n')

    tn.read_until(b'Password: ')
    tn.write(password.encode('ascii') + b'\n')

    tn.read_until(finish.encode('ascii'))

    return tn


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

    for i in range(len(ss928_ip)):

        tn = link_ss928(Host + ss928_ip[i], username, password, finish)

        # # 文件复制
        commands = [
            #  'nohup rm -rf /mnt/data/yolo/result_text/ &',
            'pkill one_camera_yolo_track_2chns_1080p; cd /mnt/data/ && sh ./run_task.sh &'

        ]
        do_telnet(tn, finish, commands)

        tn.close()
