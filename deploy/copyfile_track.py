import telnetlib
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

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

    commands = [
        'mount -t nfs -o nolock 192.168.0.77:/e/nfs_share /mnt/nfs; mkdir -p /mnt/data/sot && mkdir -p /mnt/data/one_camera_track && rm -r /mnt/data/one_camera_track/*',
        '''nohup sh -c "cp -r /mnt/nfs/one_camera_track/* /mnt/data/one_camera_track/ && cp /mnt/data/one_camera_track/run_track_task.sh /mnt/data/ && chmod +x /mnt/data/run_track_task.sh && cp /mnt/data/one_camera_track/profile /root/.profile && cp /mnt/data/run_track_task.sh /mnt/data/run_task.s" &''',

        ]

    def execute_on_device(ip):
        tn = link_ss928(Host + ip, username, password, finish)
        if tn:
            do_telnet(tn, finish, commands)
            tn.close()

    with ThreadPoolExecutor(max_workers=6) as executor:
        executor.map(execute_on_device, ss928_ip)

