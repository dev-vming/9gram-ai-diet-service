import { AuthService } from './auth.service';
import { User } from '../user/entities/user.entity';
import { Module } from '@nestjs/common';
import { AuthController } from './auth.controller';
import { GoogleStrategy } from './utils/strategies/google.strategy';
import { TypeOrmModule } from '@nestjs/typeorm';
import { SessionSerializer } from './utils/serializer';
import { UserRepository } from '../user/repositories/user.repository';
import { PassportModule } from '@nestjs/passport';
import { LocalStrategy } from './utils/strategies/local.strategy';
import { HealthInfo } from 'src/user/entities/health-info.entity';
import { HealthInfoRepository } from 'src/user/repositories/health-info.repository';
import { UserService } from 'src/user/user.service';
import { VerificationCode } from './entities/verification-code.entity';
import { VerificationCodeRepository } from './repositories/verification-code.repository';
import { MailVerificationService } from './mail-verification.service';

@Module({
  imports: [TypeOrmModule.forFeature([User, HealthInfo]), PassportModule],
  providers: [
    AuthService, 
    LocalStrategy, 
    GoogleStrategy, 
    SessionSerializer, 
    UserRepository, 
    HealthInfoRepository, 
    UserService,
    MailVerificationService,
    VerificationCodeRepository
  ],
  controllers: [AuthController]
})
export class AuthModule {}
