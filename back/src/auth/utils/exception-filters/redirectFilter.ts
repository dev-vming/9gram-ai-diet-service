import { ExceptionFilter, Catch, ArgumentsHost, UnauthorizedException, HttpException } from '@nestjs/common';
import { Request, Response } from 'express';


@Catch(HttpException)
export class ErrorToObjectFilter implements ExceptionFilter {
  catch(exception: HttpException, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();
    const request = ctx.getRequest<Request>();
    response.json({
      ...Object(exception.getResponse()),
    });
  }
}