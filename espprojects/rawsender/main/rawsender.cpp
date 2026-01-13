/**
 * @file coap_test.cpp
 * @author adapted from bluecherry_test.cpp -> Jonas Maes <jonas@dptechnics.com>
 * @date 27 Nov 2023
 * @copyright DPTechnics bv
 * @brief Walter Modem library examples
 *
 * @section LICENSE
 *
 * Copyright (C) 2023, DPTechnics bv
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1. Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *   3. Neither the name of DPTechnics bv nor the names of its contributors may
 *      be used to endorse or promote products derived from this software
 *      without specific prior written permission.
 *
 *   4. This software, with or without modification, must only be used with a
 *      Walter board from DPTechnics bv.
 *
 *   5. Any software provided in binary form under this license must not be
 *      reverse engineered, decompiled, modified and/or disassembled.
 *
 * THIS SOFTWARE IS PROVIDED BY DPTECHNICS BV “AS IS” AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL DPTECHNICS BV OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @section DESCRIPTION
 *
 * This program sends and receives mqtt data using the DPTechnics BlueCherry cloud platform.
 * It also supports OTA updates which are scheduled through the BlueCherry web interface.
 */

#include <esp_log.h>
#include <driver/uart.h>
#include "WalterModem.h"

#include "soc/gpio_struct.h"
#include "soc/sens_reg.h"
#include "soc/rtc_io_reg.h"
#include "soc/rtc_cntl_reg.h"
#include "rom/ets_sys.h"
#include "esp_sleep.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include <esp_system.h>
#include <vector>
#include <random>
// support IDF 5.x
#ifndef portTICK_RATE_MS
#define portTICK_RATE_MS portTICK_PERIOD_MS
#endif

#define GPIO_WAKEUP GPIO_NUM_9                 // connect to PPK2 Pin 0
#define GPIO_NETWORK_INITIALIZATION GPIO_NUM_1 // connect to PPK2 Pin 1
#define GPIO_SENDING GPIO_NUM_2                // connect to PPK2 Pin 2
#if CONFIG_RADIO_LTEM
#define RADIO_TECHNOLOGY WALTER_MODEM_RAT_LTEM
#endif
#if CONFIG_RADIO_NBIOT
#define RADIO_TECHNOLOGY WALTER_MODEM_RAT_NBIOT
#endif

/**
 * @brief ESP-IDF log prefixes.
 */
static constexpr const char *TIMETAG = "Timing";
static constexpr const char *MQTT_TAG = "MQTT";
static constexpr const char *COAP_TAG = "COAP";
static constexpr const char *TLS_TAG = "TLS";
static constexpr const char *TCP_TAG = "TCP";
static constexpr const char *NETWORK_TAG = "NETWORK";
// define mqtt variable "71a024e3105e41d19034a12e80c4382b.s1.eu.hivemq.cloud", 8883, "WalterClient1", "WalterESP", "kZ3NXLnY52WXEvH", 1

/**
 * @brief Cellular APN for SIM card. Leave empty to autodetect APN.
 */
CONFIG(CELLULAR_APN, const char *, "")

/**
 * @brief TLS profile used for transmission tests
 */
CONFIG_UINT8(TLS_PROFILE, 1)

/**
 * @brief COAP profile used for transmission tests
 */
CONFIG_UINT8(COAP_PROFILE, 1)

/**
 * @brief The modem instance.
 */
WalterModem modem;

/**
 * @brief Response object containing command response information.
 */
WalterModemRsp rsp = {};

/**
 * @brief Buffer for incoming COAP response
 */
uint8_t incomingBuf[256] = {0};

/**
 * @brief The counter used in the ping packets.
 */
uint16_t counter = 1;

/**
 * @brief The reason for waking up from deep sleep.
 */
u_int32_t wakeup_reason;

/**
 * @brief The binary configuration settings for PSM.
 * These can be calculated using e.g.
 * https://www.soracom.io/psm-calculation-tool/
 */
// Preferably these would allow minimal Active time, but network decides on these, so we use the one that are granted here:
// TAU (T3412): 00010010 |  Active time (T3324): 00001000
// at+cpsms=1,,,00100001,00000001
// const char *psmTAU = "00010010";  // translates to 60 seconds of TAU
const char *psmTAU = "00100001";  // translates to 60 seconds of TAU 
// const char *psmActive = "00001000"; // translates to 10 seconds of active time
const char *psmActive = "00000001"; // min active time
const char* reqEDRXVal = "1010"; // eDRX value requested by the device
const char* reqPtw =  "0000"; // Paging time window requested by the device

const char* token = "WalterModemToken";


// Returns a vector of n samples from N(0, 1)
std::vector<double> standard_normal_array(std::size_t n) {
  static std::random_device rd;
  static std::mt19937 generator(rd());
  static std::normal_distribution<double> distribution(0.0, 1.0);

  std::vector<double> samples;
  samples.reserve(n);

  for (std::size_t i = 0; i < n; ++i) {
    samples.push_back(distribution(generator));
  }

  return samples;
}
void getPSMSettings(){
  
  if(!modem.getCellInformation(WALTER_MODEM_SQNMONI_REPORTS_SERVING_CELL, &rsp)) {
    ESP_LOGI("TESTING","Error: Could not request cell information");
  } else {
    ESP_LOGI("TESTING","Connected on band %u using operator %s (%u%02u)", rsp.data.cellInformation.band,
                  rsp.data.cellInformation.netName, rsp.data.cellInformation.cc,
                  rsp.data.cellInformation.nc);
    ESP_LOGI("TESTING"," and cell ID %u.\r\n", rsp.data.cellInformation.cid);
    ESP_LOGI("TESTING","Signal strength: RSRP: %.2f, RSRQ: %.2f.\r\n", rsp.data.cellInformation.rsrp,
                  rsp.data.cellInformation.rsrq);
  }

  if (modem.sendCmd("AT+SQNINS=0", "OK")) { // Scan networks
    ESP_LOGI(NETWORK_TAG, "Successfully scanned networks");
  } else {
    ESP_LOGI(NETWORK_TAG, "Could not scan networks");
  }
}

/**
 * @brief This function checks if we are connected to the lte network
 *
 * @return True when connected, False otherwise
 */
bool lteConnected()
{
  WalterModemNetworkRegState regState = modem.getNetworkRegState();
  ESP_LOGI(NETWORK_TAG, "lteConnected check resulted in: Network registration state: %d", regState);
  return (
      regState == WALTER_MODEM_NETWORK_REG_REGISTERED_HOME ||
      regState == WALTER_MODEM_NETWORK_REG_REGISTERED_ROAMING);
}

/**
 * @brief This function waits for the modem to be connected to the Lte network.
 * @return true if the modem is connected, else false on timeout.
 */
bool waitForNetwork()
{
  /* Wait for the network to become available */
  ESP_LOGI(NETWORK_TAG, "Connecting to the %s network\n",
           RADIO_TECHNOLOGY == WALTER_MODEM_RAT_LTEM ? "LTE-M" : "NB-IoT");
  int timeout = 0;
  while (!lteConnected())
  {
    vTaskDelay(pdMS_TO_TICKS(1000));
    timeout += 1;
    if (timeout > 300)
      return false;
  }
  ESP_LOGI(NETWORK_TAG, "Connected to the network");
  return true;
}

/**
 * @brief This function tries to connect the modem to the cellular network.
 * @return true if the connection attempt is successful, else false.
 */
bool lteConnect()
{
  if (modem.setOpState(WALTER_MODEM_OPSTATE_NO_RF)) {
    ESP_LOGI(NETWORK_TAG, "Successfully set operational state to NO RF");
  } else {
    ESP_LOGI(NETWORK_TAG, "Could not set operational state to NO RF");
    return false;
  }

  /* Create PDP context */
  if (modem.definePDPContext(1, CELLULAR_APN)) {
    ESP_LOGI(NETWORK_TAG, "Created PDP context");
  } else {
    ESP_LOGI(NETWORK_TAG, "Could not create PDP context");
    return false;
  }

  /* Request PSM configuration from network */
  if (modem.configCEREGReports(WALTER_MODEM_CEREG_REPORTS_ENABLED_UE_PSM_WITH_LOCATION_EMM_CAUSE))
  {
    ESP_LOGI(NETWORK_TAG, "Configured CEREG to receive PSM result allocated by the network \r\n");
  }
  else
  {
    ESP_LOGI(NETWORK_TAG, "Could not configure CEREG to receive PSM result allocated by the network\r\n");
  }

  // modem.setOpState(WALTER_MODEM_OPSTATE_FULL);

  /* Set the operational state to full */
  if (modem.setOpState(WALTER_MODEM_OPSTATE_FULL)){
    ESP_LOGI(NETWORK_TAG, "Successfully set operational state to FULL");
  } else {
    ESP_LOGI(NETWORK_TAG, "Could not set operational state to FULL");
    return false;
  }

  /* Set the network operator selection to automatic */
  if (modem.setNetworkSelectionMode(WALTER_MODEM_NETWORK_SEL_MODE_AUTOMATIC)) {//MANUAL, "26201", WALTER_MODEM_OPERATOR_FORMAT_NUMERIC)) {
    ESP_LOGI(NETWORK_TAG, "Network selection mode to was set to automatic");
  } else {
    ESP_LOGI(NETWORK_TAG, "Could not set the network selection mode to automatic");
    return false;
  }

  return waitForNetwork();
}

/**
 * @brief Default wake stub to set GPIOs on wakeup from deep sleep
 */
void RTC_IRAM_ATTR wake_stub(void)
{
  // set RTC mux
  SET_PERI_REG_MASK(RTC_IO_TOUCH_PAD9_REG, RTC_IO_TOUCH_PAD9_MUX_SEL);
  SET_PERI_REG_MASK(SENS_SAR_PERI_CLK_GATE_CONF_REG, SENS_IOMUX_CLK_EN_M);
  // enable GPIO output
  esp_default_wake_deep_sleep();
  // Return from wake stub function to continue as usual
  return;
}

/**
 * @brief Check the wakeup cause and handle accordingly
 */
bool checkWakeupCause()
{
  wakeup_reason = esp_sleep_get_wakeup_cause();
  // check if this is a restart or wakeup
  if (wakeup_reason == ESP_SLEEP_WAKEUP_UNDEFINED) { /* Check preferred radio technology */
    ESP_LOGI("DEEPSLEEP", "Fresh start\r\n");
    if (modem.getRAT(&rsp)) {
      if (rsp.data.rat != RADIO_TECHNOLOGY) {
        modem.setRAT(RADIO_TECHNOLOGY);
        modem.reset();
        ESP_LOGI(NETWORK_TAG, "Switched modem radio technology");
      }
    } else {
      ESP_LOGE(NETWORK_TAG, "Failed to retrieve radio access technology");
      return false;
    }
    // Set PSM and eDRX settings
    
  #if CONFIG_USE_PSM
    modem.configPSM(WALTER_MODEM_PSM_ENABLE, psmTAU, psmActive);
  #else
    modem.configPSM(WALTER_MODEM_PSM_DISABLE);
  #endif
  #if CONFIG_USE_EDRX
    modem.configEDRX(WALTER_MODEM_EDRX_ENABLE, reqEDRXVal, reqPtw);
  #else
    modem.configEDRX(WALTER_MODEM_EDRX_DISABLE);
  #endif

  } else {
    ESP_LOGI("DEEPSLEEP", "Woke up from deep sleep\r\n");

    if (modem.getOpState(&rsp)) {
      ESP_LOGI("DEEPSLEEP", "Modem operational state: ");
    } else {
      ESP_LOGI("DEEPSLEEP", "Could not retrieve modem operational state");
      return false;
    }
  }
  return true;
}

bool coapReceiveConfirmation() {
    static short receiveAttemptsLeft = 200; // this variable should be reset before calling this function: receiveAttemptsLeft = 200;
    receiveAttemptsLeft = 200;
    bool messageReceived = false;
    while (receiveAttemptsLeft > 0) {
      // log number of receive attempts left
      ESP_LOGI("COAP_RING", "Waiting for COAP message, %d attempts left", receiveAttemptsLeft);
      receiveAttemptsLeft--;
      while(modem.coapDidRing(COAP_PROFILE, incomingBuf, sizeof(incomingBuf), &rsp)) {
        // log number of received COAP message attemps
        ESP_LOGW("COAP_RING", "COAP message received, %d attempts left", receiveAttemptsLeft);
        receiveAttemptsLeft = 0;
        ESP_LOGI("COAP_RING","COAP incoming:\r\n");
        ESP_LOGI("COAP_RING","profileId: %d (profile ID used by us: %d)\r\n",
          rsp.data.coapResponse.profileId, COAP_PROFILE);
        ESP_LOGI("COAP_RING","Message id: %d\r\n", rsp.data.coapResponse.messageId);
        ESP_LOGI("COAP_RING","Send type (CON, NON, ACK, RST): %d\r\n",
          rsp.data.coapResponse.sendType);
        ESP_LOGI("COAP_RING","Method or response code: %d\r\n",
          rsp.data.coapResponse.methodRsp);
        ESP_LOGI("COAP_RING","Data (%d bytes):\r\n", rsp.data.coapResponse.length);

        for(size_t i = 0; i < rsp.data.coapResponse.length; i++) {
          ESP_LOGI("COAP_RING","[%02x  %c] ", incomingBuf[i], incomingBuf[i]);
        }
        messageReceived = true;
        break;
        // modem.coapClose(COAP_PROFILE);
      }
      vTaskDelay(pdMS_TO_TICKS(25));
    }
    if (!messageReceived) {
      ESP_LOGW("COAP_RING", "No COAP message received");
      return false;
    }
    return true;
}

bool configureCOAP()
{ // TODO add your IP of the coap server
    if (modem.coapCreateContext(COAP_PROFILE, "xxx", 5000)) {
      ESP_LOGI("COAP", "Create Context succeeded");
    } else {
      ESP_LOGE("COAP", "Create Context failed");
      return false;
    }  
  if (modem.coapSetOptions(COAP_PROFILE, WALTER_MODEM_COAP_OPT_SET, WALTER_MODEM_COAP_OPT_CODE_URI_PATH, "image")){
    ESP_LOGI("COAP", "Set Options PATH succeeded");
  } else {
    ESP_LOGE("COAP", "Set Options PATH failed");
    return false;
  }

  return true;
}


/**
 * @brief Initialize GPIOs
 */
void initialize_gpios()
{
  // Set GPIOs for Wakeup to LOW
  REG_WRITE(RTC_GPIO_OUT_W1TC_REG, BIT(RTC_GPIO_OUT_DATA_W1TC_S + GPIO_WAKEUP));
  // Set GPIOs for Wakeup to OUTPUT
  gpio_set_direction(GPIO_NETWORK_INITIALIZATION, GPIO_MODE_OUTPUT);
  gpio_set_direction(GPIO_SENDING, GPIO_MODE_OUTPUT);
  // Set GPIOs for tasks to LOW
  gpio_set_level(GPIO_NETWORK_INITIALIZATION, 0);
  gpio_set_level(GPIO_SENDING, 0);
}

extern "C" void app_main(void)
{
  ESP_LOGI("APPLICATION_START", "Starting transmission application\r\n");
  initialize_gpios();

  CLEAR_PERI_REG_MASK(RTC_CNTL_PAD_HOLD_REG, BIT(GPIO_WAKEUP));
  SET_PERI_REG_MASK(RTC_IO_TOUCH_PAD9_REG, RTC_IO_TOUCH_PAD9_MUX_SEL);
  SET_PERI_REG_MASK(SENS_SAR_PERI_CLK_GATE_CONF_REG, SENS_IOMUX_CLK_EN_M);
  REG_WRITE(RTC_GPIO_ENABLE_W1TS_REG, BIT(RTC_GPIO_ENABLE_W1TS_S + GPIO_WAKEUP));
  REG_WRITE(RTC_GPIO_OUT_W1TS_REG, BIT(RTC_GPIO_OUT_DATA_W1TS_S + GPIO_WAKEUP));

  // TODO: We are waiting here to allow time for the serial monitor to connect - can we skip this? 
  const TickType_t xDelay = 3000 / portTICK_PERIOD_MS;
  vTaskDelay(xDelay);
  while(true){
    gpio_set_level(GPIO_SENDING, 1);
    std::size_t feature_number = 81;
    std::vector<double> input = standard_normal_array(feature_number);
    gpio_set_level(GPIO_SENDING, 0);

    uint32_t tStart, tNetworkInitialization, tSendImage, deltaNetworkInitialization; //, tPreprocess, tInference, tPostprocess, tEnd;
    tStart = esp_timer_get_time();
    gpio_set_level(GPIO_NETWORK_INITIALIZATION, 1);

    /* Initialize the modem */
    if (WalterModem::begin(UART_NUM_1)) {
      ESP_LOGI(NETWORK_TAG, "Successfully initialized modem");
    } else {
      ESP_LOGE(NETWORK_TAG, "Could not initialize modem");
      return;
    }

    if(modem.getRAT(&rsp)) {
      if(rsp.data.rat != RADIO_TECHNOLOGY) {
        modem.setRAT(RADIO_TECHNOLOGY);
        ESP_LOGI("TESTING","Switched modem radio technology");
      }
    } else {
      ESP_LOGI("TESTING","Error: Could not retrieve radio access technology");
    }

    if (!checkWakeupCause()) {
      ESP_LOGE(NETWORK_TAG, "Failed to check wakeup cause");
      return;
    }

    if (!lteConnected() && !lteConnect())
    {
      ESP_LOGE(NETWORK_TAG, "Unable to connect to cellular network, restarting Walter "
                            "in 10 seconds");
      vTaskDelay(pdMS_TO_TICKS(10000));
      esp_restart();
    }
    
    configureCOAP();

    tNetworkInitialization = esp_timer_get_time();
    deltaNetworkInitialization = tNetworkInitialization - tStart;
    vTaskDelay(500 / portTICK_PERIOD_MS);
    uint8_t incomingBuf[256] = { 0 };
    gpio_set_level(GPIO_NETWORK_INITIALIZATION, 0);


    gpio_set_level(GPIO_SENDING, 1);
    gpio_set_level(GPIO_NETWORK_INITIALIZATION, 1);

    static short receiveAttemptsLeft = 0;

    receiveAttemptsLeft = 200;

    modem.coapSetOptions(COAP_PROFILE, WALTER_MODEM_COAP_OPT_SET, WALTER_MODEM_COAP_OPT_CODE_URI_PATH, "image");
    modem.coapSetHeader(COAP_PROFILE, counter+1);
    
    
    uint8_t* content = reinterpret_cast<uint8_t*>(input.data());
    uint8_t content_size = input.size() * sizeof(double);
    ESP_LOGI("COAP", "Sending COAP xx with contentsize %d \r\n", content_size);
    if(modem.coapSendData(COAP_PROFILE, WALTER_MODEM_COAP_SEND_TYPE_CON, WALTER_MODEM_COAP_SEND_METHOD_PUT, content_size, content)) {
        receiveAttemptsLeft = 200;
    } 

    while(receiveAttemptsLeft > 0) {
      receiveAttemptsLeft--;
      //ESP_LOGI("COAP_RING","Checking for incoming COAP message or response\r\n");
      while(modem.coapDidRing(COAP_PROFILE, incomingBuf, sizeof(incomingBuf), &rsp)) {
        receiveAttemptsLeft = 0;
      }
      vTaskDelay(pdMS_TO_TICKS(100));
    }
    gpio_set_level(GPIO_NETWORK_INITIALIZATION, 0);
    gpio_set_level(GPIO_SENDING, 0);

    vTaskDelay(pdMS_TO_TICKS(100));
    
    esp_set_deep_sleep_wake_stub(&wake_stub);
    modem.sleep(1, true);
  }
}
