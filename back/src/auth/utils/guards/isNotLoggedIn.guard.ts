import { LocalLoginDto } from './../../dto/localLoginDto';
import { Injectable, CanActivate, ExecutionContext, UnauthorizedException, HttpStatus, HttpException } from '@nestjs/common';
import { Observable } from 'rxjs';
import { Request } from 'express';

@Injectable()
export class isNotLoggedInGuard implements CanActivate {
  canActivate(
    context: ExecutionContext,
  ): boolean | Promise<boolean> | Observable<boolean> {
    const request = context.switchToHttp().getRequest<Request>();
    if (request.user) {
      throw new HttpException("이미 로그인이 되어 있습니다.", HttpStatus.CONFLICT);
    }
    return true;
  }
}