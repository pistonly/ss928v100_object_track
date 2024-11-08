#include "tcp_tools.hpp"
#include "utils.hpp"

extern Logger logger;

TCP::~TCP() {
  if (mb_sock_connected) {
    close(m_sock);
    mb_sock_connected = false;
    mb_tcpIp_setted = false;
    logger.log(INFO, "Disconnected from TCP server.");
  }
}

int m_sock;
bool mb_sock_connected;

std::string m_tcpIp;
int m_tcpPort;
bool mb_tcpIp_setted = false;

void TCP::set_ip_port(const std::string &ip, const int port) {
  m_tcpIp = ip;
  m_tcpPort = port;
  mb_tcpIp_setted = true;
}

void TCP::connect_to_tcp() {
  struct sockaddr_in serv_addr;

  if ((m_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    logger.log(ERROR, "Socket creation failed.");
    return;
  }

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(m_tcpPort);

  if (inet_pton(AF_INET, m_tcpIp.c_str(), &serv_addr.sin_addr) <= 0) {
    logger.log(ERROR, "Invalid address / Address not supported.");
    return;
  }

  if (connect(m_sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
    logger.log(ERROR, "Connection Failed.");
    return;
  }

  mb_sock_connected = true;
  logger.log(INFO, "Connected to TCP server at ", m_tcpIp, ":", m_tcpPort);
  return;
}

void TCP::connect_to_tcp(const std::string &ip, const int port) {
  struct sockaddr_in serv_addr;

  if ((m_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    logger.log(ERROR, "Socket creation failed.");
    return;
  }

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port);

  if (inet_pton(AF_INET, ip.c_str(), &serv_addr.sin_addr) <= 0) {
    logger.log(ERROR, "Invalid address / Address not supported.");
    return;
  }

  if (connect(m_sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
    logger.log(ERROR, "Connection Failed.");
    return;
  }

  mb_sock_connected = true;
  logger.log(INFO, "Connected to TCP server at ", ip, ":", port);
  return;
}

ssize_t TCP::tcp_send(const std::vector<char> &data) {
  ssize_t bytes_sent = send(m_sock, data.data(), data.size(), MSG_NOSIGNAL);
  if (bytes_sent == -1) {
    logger.log(ERROR, "send failed");
    close(m_sock);
    mb_sock_connected = false;
  }
  return bytes_sent;
}
