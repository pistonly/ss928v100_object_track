#ifndef TCP_TOOLS_HPP
#define TCP_TOOLS_HPP

#include <arpa/inet.h>
#include <chrono>
#include <csignal> // Include this header for signal handling
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

class TCP {
public:
  TCP() : m_sock(-1), mb_sock_connected(false) {}
  ~TCP();

  int m_sock;
  bool mb_sock_connected;

  std::string m_tcpIp;
  int m_tcpPort;
  bool mb_tcpIp_setted = false;

  void set_ip_port(const std::string &ip, const int port);

  void connect_to_tcp();

  void connect_to_tcp(const std::string &ip, const int port);

  ssize_t tcp_send(const std::vector<char> &data);

private:
  std::chrono::steady_clock::time_point last_attempt_time;
};
#endif
