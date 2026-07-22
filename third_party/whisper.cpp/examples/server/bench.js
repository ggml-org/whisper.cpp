import http from 'k6/http'
import { check } from 'k6'

export let options = {
  vus: parseInt(__ENV.CONCURRENCY) || 4,
  iterations: parseInt(__ENV.CONCURRENCY) || 4,
}

const filePath        = __ENV.FILE_PATH
const baseURL         = __ENV.BASE_URL        || 'http://127.0.0.1:8080'
const endpoint        = __ENV.ENDPOINT        || '/inference'
const temperature     = __ENV.TEMPERATURE     || '0.0'
const temperatureInc  = __ENV.TEMPERATURE_INC || '0.2'
const responseFormat  = __ENV.RESPONSE_FORMAT || 'json'

// Read the file ONCE at init time
const fileBin = open(filePath, 'b')

export default function () {
  const payload = {
    file:           http.file(fileBin, filePath),
    temperature:    temperature,
    temperature_inc: temperatureInc,
    response_format: responseFormat,
  }

  const res = http.post(`${baseURL}${endpoint}`, payload)
  check(res, { 'status is 200': r => r.status === 200 })
} 