import axios, { AxiosInstance, AxiosResponse } from 'axios';

type originRequestType = {
  baseURL: string;
  headers: {
    'Content-Type': string;
    Authorization?: string;
  };
};
//AxiosRequestConfigType

const config: originRequestType = {
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
};

const config2: originRequestType = {
  baseURL: '/image',
  headers: {
    'Content-Type': 'multipart/form-data',
  },
};

export const api: AxiosInstance = axios.create(config); // 인스턴스
export const api2: AxiosInstance = axios.create(config2);

// [Client] ------[ Interceptor ] -----> [Server]
api.interceptors.request.use(
  (req) => {
    //요청 data가 formData일때
    if (req.data && req.data instanceof FormData) {
      // req.headers["Content-Type"] = "multipart/form-data";
    }
    //요청 data가 Object일 때
    else if (req.data && req.data instanceof Object) {
      req.headers['Content-Type'] = 'application/json';
    }

    return req;
  },
  (err) => {}
);

api2.interceptors.request.use(
  (req) => {
    //요청 data가 formData일때
    if (req.data && req.data instanceof FormData) {
      req.headers['Content-Type'] = 'multipart/form-data';
    }
    //요청 data가 Object일 때
    else if (req.data && req.data instanceof Object) {
      // req.headers['Content-Type'] = 'application/json';
    }

    return req;
  },
  (err) => {}
);

// [Client] <------[ Interceptor ] ----- [Server]

api.interceptors.response.use(
  (res) => {
    return res;
  },
  async (err) => {
    const { status, data } = err?.response;

    if (err.response && status === 400) {
      if (data.message[0] === '이메일 형식이 올바르지 않습니다.') {
        return data.message[0];
      }
    }

    if (err.response && status === 401) {
      if (
        data.message ===
        '등록되지 않은 이메일 이거나, 유효하지 않은 비밀번호입니다.'
      ) {
        return data.message;
      }
    }

    if (err.response && status === 409) {
      if (data.message === '이미 로컬계정으로 등록된 이메일입니다.') {
        return data.message;
      }
      if (data.message === '이미 구글계정으로 등록된 이메일입니다.') {
        return data.message;
      }
      if (data.message === '중복된 사용자명입니다.') {
        return data.message;
      }
    }
  }
);

type ApiFetcherParams = [string, any];
export type ApiMethods = 'get' | 'post' | 'put' | 'delete' | 'patch';
export type APiFetcher = (...args: ApiFetcherParams) => Promise<any>;

const getFetcher: (
  path: string,
  { params }: any
) => Promise<AxiosResponse<any, any>> = async (path, params) => {
  return await api.get(path, { params });
};
const postFetcher = async (path: string, body: any) => {
  return await api.post(path, body);
};
const patchFetcher = async (path: string, body: any) => {
  return await api.put(path, body);
};
const putFetcher = async (path: string, body: any) => {
  return await api.put(path, body);
};
const deleteFetcher = async (path: string, params: any) => {
  return await api.delete(path, { params });
};

// const args = {path:string, body: any } as const;

export const API_FETCHER: { [key in ApiMethods]: APiFetcher } = {
  get: (...args) => getFetcher(...args),
  post: (...args) => postFetcher(...args),
  put: (...args) => putFetcher(...args),
  patch: (...args) => patchFetcher(...args),
  delete: (...args) => deleteFetcher(...args),
};

export default api;
