import { ExceptionFilter, Catch, ArgumentsHost, UnauthorizedException, HttpException } from '@nestjs/common';
import { Request, Response } from 'express';

@Catch(HttpException)
export class CustomErrorFilter implements ExceptionFilter {
  catch(exception: HttpException, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();
    const request = ctx.getRequest<Request>();

        if(exception.getStatus() === 400){
            response.status(400).send('잘못된 요청입니다.');
        }else if(exception.getStatus() === 401){
            response.status(401).send('로그인이 필요합니다.');
        }else if (exception.getStatus() === 402){
            response.status(402).send('결제가 필요합니다.');
        }else if(exception.getStatus() === 403){
            response.status(403).send('권한이 없습니다.');    
        }else if (exception.getStatus() === 404){
            response.status(404).send('찾을 수 없는 페이지입니다.');
        }else if( exception.getStatus() === 405){
            response.status(405).send('허용되지 않은 메서드입니다.');
        }else if(exception.getStatus() === 404){
            response.status(404).send('찾을 수 없는 페이지입니다.');
        }else if(exception.getStatus() === 409){
            response.status(409).send('이미 존재하는 데이터입니다.');
        }else if(exception.getStatus() === 500){
            response.status(500).send('서버에 오류가 발생했습니다.');
        }else if(exception.getStatus() === 501){
            response.status(501).send('서버에서 지원하지 않는 요청입니다.');
        }else if (exception.getStatus() === 502){
            response.status(502).send('서버와의 연결이 끊어졌습니다.');
        }else if (exception.getStatus() === 503){
            response.status(503).send('서버가 점검중입니다.');
        }else if (exception.getStatus() === 504){
            response.status(504).send('서버와의 연결이 지연되고 있습니다.');
        }else if (exception.getStatus() === 505){
            response.status(505).send('HTTP 버전이 지원되지 않습니다.');
        }else if (exception.getStatus() === 507){
            response.status(507).send('서버의 저장공간이 부족합니다.');
        }

        if (exception.getStatus.toString().startsWith('5')) {
            throw exception;
        }else {
            console.log(exception);
        }
    }
}