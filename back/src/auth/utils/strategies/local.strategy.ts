import { Strategy } from 'passport-local';
import { PassportStrategy } from '@nestjs/passport';
import { HttpException, HttpStatus, Injectable, UnauthorizedException } from '@nestjs/common';
import { AuthService } from '../../auth.service';

@Injectable()
export class LocalStrategy extends PassportStrategy(Strategy) {
  constructor(private authService: AuthService) {
    super({ usernameField: 'email', passwordField: 'password' }); 
  }

  async validate(email: string, password: string, done: Function): Promise<any> {
    try{
      if(!email || !password) throw new HttpException("등록되지 않은 이메일 이거나, 유효하지 않은 비밀번호입니다.", HttpStatus.UNAUTHORIZED);
      const user = await this.authService.validateLocalUser(email, password);
      if (!user) {
        throw new HttpException("등록되지 않은 이메일 이거나, 유효하지 않은 비밀번호입니다.", HttpStatus.UNAUTHORIZED);
      }
      return done(null, user);
    }catch(err){
      return done(err, false);
    }
  }
}




