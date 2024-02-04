import { ExecutionContext, HttpException, HttpStatus } from "@nestjs/common";
import { AuthGuard } from "@nestjs/passport";
import { LocalLoginDto } from "src/auth/dto/localLoginDto";

export class LocalAuthGuard extends AuthGuard('local') {
  async canActivate(context: ExecutionContext) {
    try{
      const request = context.switchToHttp().getRequest();
      const localLoginDto: LocalLoginDto = request.body;
      if(!localLoginDto.email || !localLoginDto.password) throw new HttpException("등록되지 않은 이메일 이거나, 유효하지 않은 비밀번호입니다.", HttpStatus.UNAUTHORIZED);
      const activate = (await super.canActivate(context)) as boolean; // 로그인 성공시 true, 실패시 false
      await super.logIn(request); // 세션에 사용자 정보를 저장
      return activate;
    }catch(err){
      throw err;
    }
  }
}