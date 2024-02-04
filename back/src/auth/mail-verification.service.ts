import { MailerModule, MailerService } from '@nestjs-modules/mailer';
import { VerificationCode } from './entities/verification-code.entity';
import { VerificationCodeRepository } from './repositories/verification-code.repository';
import { DataSource } from 'typeorm';
import { Injectable } from '@nestjs/common';

@Injectable()
export class MailVerificationService {
    constructor(
        private readonly verificationCodeRepository : VerificationCodeRepository,
        private readonly mailerService: MailerService,
        private readonly dataSource: DataSource
    ){}

    async sendVerificationCode(email: string){
        const queryRunner = this.dataSource.createQueryRunner();
        await queryRunner.connect();
        await queryRunner.startTransaction();
        try{
            // 인증코드 생성
            const code = Math.floor(Math.random() * 10000).toString();
            // 인증코드 객체 생성
            const verificationCode = new VerificationCode(email, code);
            //인증코드 저장
            await this.verificationCodeRepository.saveVerificationCode(verificationCode, queryRunner.manager);
            
            // 이메일 전송
            const result = await this.mailerService.sendMail({
                to: email,
                subject: 'gugram 이메일 인증 코드',
                html: `<h1>gugram 회원가입 이메일 인증 코드</h1><h3>${code}</h3>`
            });
            
            await queryRunner.commitTransaction();
            return result;
        }catch(err){ 
            await queryRunner.rollbackTransaction();
            throw err; 
        }finally{ await queryRunner.release(); }
    }

    async verifyCode(email: string, code: string){
        const queryRunner = this.dataSource.createQueryRunner();
        await queryRunner.connect();
        await queryRunner.startTransaction();
        try{
            // 인증코드 조회
            const verificationCode = await this.verificationCodeRepository.findVerificationCodeByEmail(email, queryRunner.manager);
            // 인증코드가 없으면
            if(!verificationCode){
                await queryRunner.commitTransaction();
                return false;
            }
            // 인증코드가 일치하지 않으면
            if(verificationCode.code !== code){
                await queryRunner.commitTransaction();
                return false;
            }
            // 인증코드 삭제
            await this.verificationCodeRepository.deleteVerificationCodeByEmail(email, queryRunner.manager);
            await queryRunner.commitTransaction();
            return true;
        }catch(err){ 
            await queryRunner.rollbackTransaction();
            throw err; 
        }finally{ await queryRunner.release(); }
    }
}